'''
Incoporate accumulate for multi-gpu training and diffusers for training diffusion models.
Author: Rundong Luo
'''
import os
from collections import OrderedDict
from functools import partial

from utils.util_3d import sdf_to_mesh_trimesh
import numpy as np
import mcubes
from omegaconf import OmegaConf
from termcolor import colored, cprint
from einops import rearrange, repeat
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn, optim

import torchvision.utils as vutils
import torchvision.transforms as transforms

from models.base_model import BaseModel
from models.networks.vqvae_networks.network import VQVAE
from models.networks.diffusion_networks.network import DiffusionUNet
from models.model_utils import load_vqvae
from models.networks.ply_networks.pointnet2 import PointNet2
from models.networks.ply_networks.pointnet import PointNetEncoder

from diffusers import DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

from accelerate import Accelerator

from utils.util import AverageMeter
from datasets.mesh_to_sdf import mesh_to_sdf

from planners.base_model import create_planner

# rendering
from utils.util_3d import init_mesh_renderer, render_sdf

class SDFusionModelPly2ShapeRefineAcc(BaseModel):
    def name(self):
        return 'SDFusion-Model-PointCloud-to-Shape-Refinement-Accelerate'

    def __init__(self, opt, accelerator: Accelerator, input_instance: dict):
        super().__init__(opt)
        self.isTrain = opt.isTrain
        self.model_name = self.name()
        self.device = accelerator.device
        self.accelerator = accelerator
        assert self.opt.ply_cond

        # self.optimizer_skip = False
        ######## START: Define Networks ########
        assert opt.df_cfg is not None
        assert opt.vq_cfg is not None

        # init df
        self.df_conf = df_conf = OmegaConf.load(opt.df_cfg)
        self.vq_conf = vq_conf = OmegaConf.load(opt.vq_cfg)

        # record z_shape
        ddconfig = vq_conf.model.params.ddconfig
        self.shape_res = ddconfig.resolution
        z_ch, n_down = ddconfig.z_channels, len(ddconfig.ch_mult)-1
        z_sp_dim = self.shape_res // (2 ** n_down)
        self.z_shape = (z_ch, z_sp_dim, z_sp_dim, z_sp_dim)
        # init diffusion networks
        unet_params = df_conf.unet.params
        self.df = DiffusionUNet(unet_params, vq_conf=vq_conf, conditioning_key=df_conf.model.params.conditioning_key).to(self.device)
        self.df.requires_grad_(False)
        self.guidance_scale = opt.uc_scale

        # init vqvae
        self.vqvae = load_vqvae(vq_conf, vq_ckpt=opt.vq_ckpt, opt=opt).to(self.device)
        self.vqvae.requires_grad_(False)

        # init U-Net conditional model
        self.cond_model = PointNet2(hidden_dim=df_conf.unet.params.context_dim).to(self.device)
        # convert to sync-bn
        self.cond_model = nn.SyncBatchNorm.convert_sync_batchnorm(self.cond_model)
        self.cond_model.requires_grad_(False)

        # load pre-trained weights
        self.load_ckpt(ckpt=opt.pretrained_ckpt)

        # noise scheduler
        self.noise_scheduler = DDIMScheduler()

        self.x = input_instance['sdf'].to(self.device) # (1, 1, res, res, res)
        self.ply = input_instance['ply'].to(self.device) # (1, 3, N)

        if 'initial_shape' in input_instance:
            self.initial_shape = input_instance['initial_shape'].to(self.device)
            with torch.no_grad():
                latent = self.vqvae.encode_no_quant(self.initial_shape)
        else:
            save_filename = 'latent_%s.pth' % (opt.model_id)
            save_path = os.path.join(opt.ckpt_dir, save_filename)
            if accelerator.is_main_process:
                latent = self.inference()
                torch.save(latent.detach().cpu(), save_path) # save the latent
            accelerator.wait_for_everyone() # load the latent for all processes
            latent = torch.load(save_path, map_location=self.device)
        
        self.latent = nn.Parameter(latent, requires_grad=True) # (1, latent_dim, vq_res, vq_res, vq_res)
        self.paths = input_instance['path']
        
        # transformation info for calculating collision loss
        # "ply_points + ply_translation" aligns with "part * part_extent_ratio + part_translation"
        self.ply_translation = input_instance['ply_translation'].to(self.device) # (1, 3)
        self.part_translation = input_instance['part_translation'].to(self.device) # (1, 3)
        self.part_extent = input_instance['part_extent'].to(self.device) # (1, 3)

        if self.opt.use_mobility_constraint:
            self.move_axis = input_instance['move_axis'].to(self.device) # (3,)
            self.move_limit = input_instance['move_limit'].to(self.device) # (2,) range of motion

        self.batch_size = opt.batch_size

        if self.isTrain:
            self.optimizer = optim.AdamW([self.latent], lr=opt.lr)

            lr_lambda = lambda it: 0.5 * (1 + np.cos(np.pi * it / opt.total_iters))
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

            self.optimizers = [self.optimizer]
            self.schedulers = [self.scheduler]

            self.print_networks(verbose=False)

        # prepare accelerate
        self.df, self.vqvae, self.cond_model = accelerator.prepare(self.df, self.vqvae, self.cond_model)

        ######## END: Define Networks ########

        # setup renderer
        if 'snet' in opt.dataset_mode:
            dist, elev, azim = 1.7, 20, 120
        elif 'pix3d' in opt.dataset_mode:
            dist, elev, azim = 1.7, 20, 120
        elif opt.dataset_mode == 'buildingnet':
            dist, elev, azim = 1.0, 20, 120
        elif opt.dataset_mode == 'gapnet':
            dist, elev, azim = 1.0, 20, 120

        self.renderer = init_mesh_renderer(image_size=256, dist=dist, elev=elev, azim=azim, device=self.device)

        self.use_collision_loss = opt.collision_loss
        self.loss_collision_weight = opt.loss_collision_weight
        self.loss_names = ['sds', 'collision', 'total']
        self.loss_meter_dict = {name: AverageMeter() for name in self.loss_names}
        self.loss_meter_epoch_dict = {name: AverageMeter() for name in self.loss_names}

    def set_input(self, input=None, max_sample=None):
        raise NotImplementedError('set_input() is not implemented')

    def switch_train(self):
        raise NotImplementedError('switch_train() is not implemented')

    def switch_eval(self):
        self.df.eval()
        self.cond_model.eval()
        self.vqvae.eval()

    # check: ddpm.py, line 891
    def apply_model(self, x_noisy, t, cond, return_ids=False):

        """
            self.model = DiffusionWrapper(unet_config, conditioning_key)
            => which is the denoising UNet.
        """

        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            # key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            key = 'c_concat' if self.df_conf.model.params.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        # eps
        out = self.df(x_noisy, t, **cond)

        if isinstance(out, tuple) and not return_ids:
            return out[0]
        else:
            return out
    
    def get_sdf_refinement_loss(self):
        '''
        latent: the latent representation of the input sdf, currently bsz = 1, expand before feeding into the model
        calculate the sdf refinement loss
        including: sds loss and collision loss
        return the total loss
        '''

        self.switch_eval()
        loss_dict = {}

        latents = self.latent.expand(self.batch_size, -1, -1, -1, -1) # (B, latent_dim, res, res, res)
        ply = self.ply.expand(self.batch_size, -1, -1) # (B, 3, N)
        
        # 1. sds loss
        B = self.batch_size
        c = self.cond_model(ply).unsqueeze(1) # (B, 1, context_dim)
        uc = self.cond_model(uncond=True).unsqueeze(0).unsqueeze(0).repeat(B, 1, 1)
        c_full = torch.cat([uc, c])
        
        # sample timesteps
        min_noise = 0.02 * self.noise_scheduler.config.num_train_timesteps
        max_noise = max(0.98 * (1 - self.scheduler.last_epoch / self.opt.total_iters), 0.5) * self.noise_scheduler.config.num_train_timesteps
        timesteps = torch.randint(
            int(min_noise), int(max_noise), (B,), device=latents.device,
            dtype=torch.int64
        )

        noise = torch.randn(latents.shape, device=latents.device)
        # Add noise to the clean images according to the noise magnitude at each timestep
        latent_noisy = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = self.apply_model(torch.cat([latent_noisy] * 2), torch.cat([timesteps]*2), c_full)

        # perform guidance
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

        weight = self.noise_scheduler.alphas_cumprod[timesteps].view(B, 1, 1, 1, 1)

        grad = (noise_pred - noise) * weight
        target = (latents - grad).detach()

        loss_sds = F.mse_loss(latents, target, reduction="mean")
        
        loss_dict['sds'] = loss_sds

        # 2. collision loss
        if self.use_collision_loss:
            B = 1 if not self.opt.use_mobility_constraint else self.opt.mobility_sample_count
            # 0) decode the latents to sdf
            sdf = self.vqvae.decode(self.latent).expand(B, -1, -1, -1, -1) # (1, 1, res_sdf, res_sdf, res_sdf)

            with torch.no_grad():
                # 1) build mesh from sdf (latents), not differentiable
                # mesh = sdf_to_mesh_trimesh(sdf[0][0], spacing=(2./self.shape_res, 2./self.shape_res, 2./self.shape_res))
                # scale = torch.max(self.part_extent) / np.max(mesh.extents)
                scale = torch.max(self.part_extent) / (4 / 2.2)
                # 2) transform point cloud to the mesh coordinate
                ply_transformed = (self.ply + self.ply_translation.view(1, 3, 1) - self.part_translation.view(1, 3, 1)) / scale # (1, 3, N)
                # 3) if use mobility constraint, apply the constraint, randomly sample a distance
                if self.opt.use_mobility_constraint:
                    dist = torch.rand([B, 1, 1], device=self.device) * (self.move_limit[1] - self.move_limit[0]) + self.move_limit[0] # (B, 1, 1)
                    dist_vec = self.move_axis.view(1, 3, 1) * dist.repeat(1, 3, 1) # (B, 3, 1)
                    ply_transformed = ply_transformed.expand(B, -1, -1) - dist_vec # (B, 3, N) move the part, i.e., reversely move the point cloud
                ply_transformed = ply_transformed.transpose(1, 2) # (B, N, 3)
            
            # 3) query the sdf value at the transformed point cloud
            # input: (1, 1, res_sdf, res_sdf, res_sdf), (B, 1, 1, N, 3) -> (B, 1, 1, 1, N)
            sdf_ply = F.grid_sample(sdf, ply_transformed.unsqueeze(1).unsqueeze(1), align_corners=True).squeeze(1).squeeze(1).squeeze(1) # (B, N)
            # 4) calculate the collision loss
            loss_collision = torch.sum(torch.max(F.relu(-sdf_ply), dim=0)[0]) # (B, N) -> (B, 1) -> scalar
            # loss_collision = torch.sum(F.relu(-sdf_ply-0.001)) / B # (B, N) -> (B, 1) -> scalar
            # loss_collision = torch.mean(F.relu(-sdf_ply)) # (B, N) -> (B, 1) -> scalar 
            loss_collision_weight = self.loss_collision_weight * max(0, 2 * self.scheduler.last_epoch / self.opt.total_iters - 1)
            # loss_collision_weight = self.loss_collision_weight
            loss_dict['collision'] = loss_collision * loss_collision_weight
        else:
            loss_dict['collision'] = torch.tensor(0.0, device=self.device)
        
        loss_dict['total'] = loss_dict['sds'] + loss_dict['collision']

        return loss_dict
    
    def forward(self):

        self.switch_eval()
        loss_dict = self.get_sdf_refinement_loss()

        return loss_dict

    def backward(self): # not used
        raise NotImplementedError('backward() is not implemented')

    def optimize_parameters(self, total_steps):
        # self.set_requires_grad([self.df], requires_grad=True)

        loss_dict = self.forward()
        avg_loss_dict = {k: self.accelerator.gather(v).mean() for k, v in loss_dict.items()}
        for k, v in avg_loss_dict.items():
            self.loss_meter_dict[k].update(v.item(), self.opt.batch_size)
            self.loss_meter_epoch_dict[k].update(v.item(), self.opt.batch_size)
        self.accelerator.backward(loss_dict['total'])
        
        # clip grad norm
        torch.nn.utils.clip_grad_norm_(self.latent, 0.1)
        
        for optimizer in self.optimizers:
            optimizer.step()
        
        for scheduler in self.schedulers:
            scheduler.step()

        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def get_current_errors(self):
        
        ret = OrderedDict([(k, v.avg) for k, v in self.loss_meter_dict.items()])

        for _, v in self.loss_meter_dict.items():
            v.reset()

        # print learning rate
        for i, scheduler in enumerate(self.schedulers):
            print(f'learning rate at iter {scheduler.last_epoch}: {scheduler.get_last_lr()}', flush=True)

        return ret
    
    @torch.no_grad()
    def eval_metrics(self, dataloader, thres=0.0, global_step=0):

        ret = OrderedDict([(k, v.avg) for k, v in self.loss_meter_epoch_dict.items()])

        for _, v in self.loss_meter_epoch_dict.items():
            v.reset()

        return ret

    def get_current_visuals(self):

        with torch.no_grad():
            self.img_gt = render_sdf(self.renderer, self.x)
            self.gen_df = self.vqvae.decode(self.latent)
            self.img_gen_df = render_sdf(self.renderer, self.gen_df)
            spc = (2./self.shape_res, 2./self.shape_res, 2./self.shape_res)
            meshes = [sdf_to_mesh_trimesh(self.gen_df[i][0], spacing=spc) for i in range(self.gen_df.shape[0])]

        vis_tensor_names = [
            'img_gt',
            'img_gen_df',
        ]
        vis_ims = self.tnsrs2ims(vis_tensor_names)
        visuals = zip(vis_tensor_names, vis_ims)
        visuals_dict = {
            "img": OrderedDict(visuals),
            "meshes": meshes,
            "paths": self.paths,
            "points": self.ply.cpu().numpy(),
            "sdf": self.gen_df.cpu().numpy(),
        }

        if hasattr(self, 'ply_translation'):
            visuals_dict['ply_translation'] = self.ply_translation.cpu().numpy()
            visuals_dict['part_translation'] = self.part_translation.cpu().numpy()
            mesh_extents = torch.zeros([0, 3], device=self.device)
            for mesh in meshes:
                mesh_extents = torch.cat([mesh_extents, torch.tensor(mesh.extents, device=self.device).unsqueeze(0)], dim=0)
            visuals_dict['part_scale'] = (torch.max(self.part_extent, dim=1)[0] / (4 / 2.2)).cpu().numpy()
            # visuals_dict['part_scale'] = (torch.max(self.part_extent, dim=1)[0] / torch.max(mesh_extents, dim=1)[0]).cpu().numpy()

        return visuals_dict

    def save(self, label, global_step, save_opt=False):

        state_dict = {
            'latent': self.latent,
        }

        for i, optimizer in enumerate(self.optimizers):
            state_dict[f'opt{i}'] = optimizer.state_dict()
        for i, scheduler in enumerate(self.schedulers):
            state_dict[f'sch{i}'] = scheduler.state_dict()
           
        save_filename = 'latent_%s.pth' % (label)
        save_path = os.path.join(self.opt.ckpt_dir, save_filename)

        torch.save(state_dict, save_path)

    def load_ckpt(self, ckpt, load_opt=False):
        map_fn = lambda storage, loc: storage
        if type(ckpt) == str:
            state_dict = torch.load(ckpt, map_location=map_fn)
        else:
            state_dict = ckpt

        self.df.load_state_dict(state_dict['df'])
        print(colored('[*] weight successfully load from: %s' % ckpt, 'blue'))

        # if 'opt' in state_dict:
        for i, optimizer in enumerate(self.optimizers):
            try:
                optimizer.load_state_dict(state_dict[f'opt{i}'])
                print(colored('[*] optimizer successfully restored from: %s' % ckpt, 'blue'))
            except:
                print(colored('[*] optimizer not loaded', 'red'))
        
        iter_passed = state_dict['global_step']
        
        if 'sch' in state_dict:
            for i, scheduler in enumerate(self.schedulers):
                try:
                    scheduler.load_state_dict(state_dict[f'sch{i}'])
                    print(colored('[*] scheduler successfully restored from: %s' % ckpt, 'blue'))
                except:
                    print(colored('[*] scheduler not loaded', 'red'))
                    for _ in range(state_dict['global_step']):
                        scheduler.step()

        return iter_passed
    
    @torch.no_grad()
    def inference(self, ddim_steps=100, eta=0.):

        self.switch_eval()
            
        B = 1
        shape = self.z_shape
        c = self.cond_model(self.ply).unsqueeze(1) # (B, context_dim), point cloud condition
        uc = self.cond_model(uncond=True).unsqueeze(0).unsqueeze(0).repeat(B, 1, 1)
        c_full = torch.cat([uc, c])

        latents = torch.randn((B, *shape), device=self.device)
        latents = latents * self.noise_scheduler.init_noise_sigma

        self.noise_scheduler.set_timesteps(ddim_steps)

        # w/ condition
        for t in tqdm(self.noise_scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, timestep=t)

            # predict the noise residual
            with torch.no_grad():
                timesteps = torch.full((B*2,), t, device=self.device, dtype=torch.int64)
                noise_pred = self.apply_model(latent_model_input, timesteps, c_full)

            # perform guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.noise_scheduler.step(noise_pred, t, latents, eta).prev_sample

        return latents


'''
Incoporate diffusers for training diffusion models.
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
from models.utils import get_collision_loss

from diffusers import DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

from planners.base_model import create_planner

from utils.util import AverageMeter

# rendering
from utils.util_3d import init_mesh_renderer, render_sdf

class SDFusionModelPly2Shape(BaseModel):
    def name(self):
        return 'SDFusion-Model-PointCloud-to-Shape'

    def __init__(self, opt):
        super().__init__(opt)
        self.isTrain = opt.isTrain
        self.model_name = self.name()
        self.device = 'cuda'
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
        self.df = DiffusionUNet(unet_params, vq_conf=vq_conf, conditioning_key=df_conf.model.params.conditioning_key)
        self.df.to(self.device)
        self.parameterization = "eps"
        self.guidance_scale = opt.uc_scale
        # self.init_diffusion_params(scale=1, opt=opt)

        # init vqvae
        self.vqvae = load_vqvae(vq_conf, vq_ckpt=opt.vq_ckpt, opt=opt).to(self.device)

        # init U-Net conditional model
        self.cond_model = PointNet2(hidden_dim=df_conf.unet.params.context_dim).to(self.device)
        self.cond_model.requires_grad_(True)
        load_result = self.cond_model.load_state_dict(torch.load(opt.cond_ckpt)['model_state_dict'], strict=False)
        print(load_result)
        print(colored('[*] conditional model successfully loaded', 'blue'))
        self.uncond_prob = df_conf.model.params.uncond_prob

        ######## END: Define Networks ########

        if self.isTrain:
            # initialize optimizers
            # self.optimizer1 = optim.AdamW([p for p in self.df.parameters() if p.requires_grad == True], lr=opt.lr)
            # self.optimizer2 = optim.AdamW([p for p in self.cond_model.parameters() if p.requires_grad == True], lr=opt.lr)

            # lr_lambda1 = lambda it: 0.5 * (1 + np.cos(np.pi * it / opt.total_iters))
            # self.scheduler1 = optim.lr_scheduler.LambdaLR(self.optimizer1, lr_lambda1)
            
            # freeze_iters = 10000
            # lr_lambda2 = lambda it: 0.5 * (1 + np.cos(np.pi * it / opt.total_iters)) if it > freeze_iters else 0
            # self.scheduler2 = optim.lr_scheduler.LambdaLR(self.optimizer2, lr_lambda2)

            # self.optimizers = [self.optimizer1, self.optimizer2]
            # self.schedulers = [self.scheduler1, self.scheduler2]

            self.optimizer1 = optim.AdamW([p for p in self.df.parameters() if p.requires_grad == True] + \
                            [p for p in self.cond_model.parameters() if p.requires_grad == True], lr=opt.lr)

            lr_lambda1 = lambda it: 0.5 * (1 + np.cos(np.pi * it / opt.total_iters))
            self.scheduler1 = optim.lr_scheduler.LambdaLR(self.optimizer1, lr_lambda1)

            self.optimizers = [self.optimizer1]
            self.schedulers = [self.scheduler1]

            self.print_networks(verbose=False)

            if opt.continue_train:
                self.start_iter = self.load_ckpt(ckpt=os.path.join(opt.ckpt_dir, f'df_steps-{opt.load_iter}.pth'))
            else:
                self.start_iter = 0

        else:
            self.load_ckpt(ckpt=os.path.join(opt.ckpt_dir, f'df_steps-{opt.load_iter}.pth'))

        # noise scheduler
        self.noise_scheduler = DDIMScheduler()

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

        self.ddim_steps = self.df_conf.model.params.ddim_steps
        if self.opt.debug == "1":
            # NOTE: for debugging purpose
            self.ddim_steps = 7
        cprint(f'[*] setting ddim_steps={self.ddim_steps}', 'blue')
        self.planner = None
        if not self.isTrain:
            self.planner = create_planner(opt)

        self.loss_meter = AverageMeter()
        self.loss_meter.reset()
        self.loss_meter_epoch = AverageMeter()
        self.loss_meter_epoch.reset()
    
    def set_input(self, input=None, max_sample=None, transform_info=False):
        
        self.x = input['sdf'].to(self.device)
        self.ply = input['ply'].to(self.device)
        self.paths = input['path']

        # transformation info for calculating collision loss
        # "ply_points + ply_translation" aligns with "part * part_extent + part_translation"
        if transform_info:
            self.ply_translation = input['ply_translation'].to(self.device)
            self.part_translation = input['part_translation'].to(self.device)
            self.part_extent = input['part_extent'].to(self.device)

        if self.opt.use_mobility_constraint:
            self.move_axis = input['move_axis'].to(self.device) # (3,)
            self.move_limit = input['move_limit'].to(self.device) # (2,) range of motion
        else:
            self.move_axis = None
            self.move_limit = None

    def switch_train(self):
        self.df.train()
        self.cond_model.train()
        self.vqvae.eval()

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
            key = 'c_concat' if self.df.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        # eps
        out = self.df(x_noisy, t, **cond)

        if isinstance(out, tuple) and not return_ids:
            return out[0]
        else:
            return out

    def get_loss(self, pred, target, loss_type='l2', mean=True):
        if loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def forward(self):

        self.switch_train()

        B = self.x.shape[0]
        c = self.cond_model(self.ply).unsqueeze(1) # (B, context_dim)
        uc = self.cond_model(uncond=True).unsqueeze(0).unsqueeze(0).repeat(B, 1, 1) # (B, 1, context_dim), unconditional condition
        # drop cond with self.uncond_prob
        c = torch.where(torch.rand(B, 1, device=self.device) < self.uncond_prob, uc, c)

        # 1. encode to latent
        #    encoder, quant_conv, but do not quantize
        with torch.no_grad():
            z = self.vqvae(self.x, forward_no_quant=True, encode_only=True)

        # 2. do diffusion's forward
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (z.shape[0],), device=z.device,
            dtype=torch.int64
        )
        noise = torch.randn(z.shape, device=z.device)
        # Add noise to the clean images according to the noise magnitude at each timestep
        z_noisy = self.noise_scheduler.add_noise(z, noise, timesteps)
        model_output = self.apply_model(z_noisy, timesteps, cond=c)

        if self.parameterization == "x0":
            target = z
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        loss = self.get_loss(model_output, target).mean()
        return loss

    @torch.no_grad()
    def inference(self, data, sample=True, ddim_steps=None, ddim_eta=0., quantize_denoised=True,
                  infer_all=False, max_sample=16, transform_info=False):

        self.switch_eval()
        
        self.set_input(data, transform_info=transform_info)
        
        if ddim_steps is None:
            ddim_steps = self.ddim_steps
            
        B = self.opt.batch_size
        shape = self.z_shape
        c = self.cond_model(self.ply).unsqueeze(1) # (B, context_dim), point cloud condition
        uc = uc = self.cond_model(uncond=True).unsqueeze(0).unsqueeze(0).repeat(B, 1, 1)
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
            latents = self.noise_scheduler.step(noise_pred, t, latents, eta=ddim_eta).prev_sample

        # decode z
        self.gen_df = self.vqvae.decode_no_quant(latents)

        if self.opt.print_collision_loss:
            for i in range(B):
                gen_sdf_i = self.gen_df[i:i+1].repeat(32, 1, 1, 1, 1)
                collision_loss = get_collision_loss(gen_sdf_i, self.ply[i:i+1], self.ply_translation[i:i+1], 
                                                    self.part_extent[i:i+1], self.part_translation[i:i+1],
                                                    move_limit=self.move_limit[i], move_axis=self.move_axis[i])
                print(f'Collision Loss for Instance {i}:', collision_loss, '\n')

    def guided_inference(self, data, ddim_steps=None, ddim_eta=0., n_sample_x0=1, transform_info=False):
        
        self.switch_eval()

        self.set_input(data, transform_info=transform_info)

        if ddim_steps is None:
            ddim_steps = self.ddim_steps
        else:
            self.ddim_steps = ddim_steps

        B = self.x.shape[0]
        assert B == 1 # only support batch size 1 for now
        shape = self.z_shape
        c = self.cond_model(self.ply).unsqueeze(1) # (B, context_dim), point cloud condition
        uc = self.cond_model(uncond=True).unsqueeze(0).unsqueeze(0).repeat(B, 1, 1)
        c_full = torch.cat([uc, c])

        latents = torch.randn((B, *shape), device=self.device) # (B, *shape)
        latents = latents * self.noise_scheduler.init_noise_sigma

        self.noise_scheduler.set_timesteps(ddim_steps)

        # w/ condition
        for i, t in tqdm(enumerate(self.noise_scheduler.timesteps)):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, timestep=t)

            # predict the noise residual
            with torch.no_grad():
                timesteps = torch.full((B*2,), t, device=self.device, dtype=torch.int64)
                noise_pred = self.apply_model(latent_model_input, timesteps, c_full)

            # perform guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_cond

            # compute the previous noisy sample x_t -> x_t-1
            with torch.enable_grad():
                latents_grad = latents.detach().requires_grad_(True)
                pred_x0 = self.noise_scheduler.step(noise_pred, t, latents_grad, eta=ddim_eta).pred_original_sample

                setattr(self, f'pred_sdf_x0_{i}', self.vqvae.decode_no_quant(pred_x0).detach())
                
                # add noise to pred_x0, with std = sigma_t / sqrt(1 + sigma_t^2)
                # prev_t = t - self.noise_scheduler.config.num_train_timesteps // self.noise_scheduler.num_inference_steps
                # sigma_t = self.noise_scheduler._get_variance(t, prev_t) ** 0.5
                # noise = torch.randn([n_sample_x0, *shape], device=self.device) * (sigma_t / (1 + sigma_t ** 2) ** 0.5) # (n_sample_x0, *shape)
                # pred_x0_noisy = pred_x0.expand(n_sample_x0, -1, -1, -1, -1) + noise
                
                pred_x0_noisy = pred_x0.expand(n_sample_x0, -1, -1, -1, -1)

                pred_x0_noisy_sdf = self.vqvae.decode(pred_x0_noisy)

                collision_loss = get_collision_loss(pred_x0_noisy_sdf, self.ply, self.ply_translation, self.part_extent, self.part_translation,
                                                    move_limit=self.move_limit[0], move_axis=self.move_axis[0], loss_collision_weight=1)
                print('Collision Loss:', collision_loss, '\n')
                
                if i >= ddim_steps // 2:
                    grad = torch.autograd.grad(collision_loss, latents_grad)[0] # (B, *shape)
                    grad = 20 * grad / (grad.norm() + 1e-8) # clip grad norm
                    noise_pred = noise_pred + (1 - self.noise_scheduler.alphas_cumprod[t]) ** 0.5 * grad
            
            noise_pred = noise_pred + (self.guidance_scale - 1) * (noise_pred_cond - noise_pred_uncond)

            latents = self.noise_scheduler.step(noise_pred, t, latents, eta=ddim_eta).prev_sample

        # decode z
        self.gen_df = self.vqvae.decode_no_quant(latents)

    @torch.no_grad()
    def eval_metrics(self, dataloader, thres=0.0, global_step=0):
        
        ret = OrderedDict([
            ('loss', self.loss_meter_epoch.avg),
        ])
        self.loss_meter_epoch.reset()

        return ret

    def backward(self): # not used
        self.loss.backward()

    def optimize_parameters(self, total_steps):
        # self.set_requires_grad([self.df], requires_grad=True)

        loss = self.forward()
        avg_loss = self.accelerator.gather(loss).mean()
        self.loss_meter.update(avg_loss, self.opt.batch_size)
        self.loss_meter_epoch.update(avg_loss, self.opt.batch_size)
        loss.backward()
        
        # clip grad norm
        torch.nn.utils.clip_grad_norm_(self.df.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.cond_model.parameters(), 1.0)
        
        for optimizer in self.optimizers:
            optimizer.step()
        
        for scheduler in self.schedulers:
            scheduler.step()

        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def get_current_errors(self):
        
        ret = OrderedDict([
            ('total', self.loss_meter.avg),
        ])

        self.loss_meter.reset()

        return ret

    def get_current_visuals(self):

        with torch.no_grad():
            self.img_gt = render_sdf(self.renderer, self.x)
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
            "points": self.ply.cpu().numpy(), # (B, 3, N)
        }

        if hasattr(self, f'pred_sdf_x0_{self.ddim_steps-1}'):
            meshes_pred = [sdf_to_mesh_trimesh(getattr(self, f'pred_sdf_x0_{self.ddim_steps-1}')[0][0], spacing=spc) for i in range(self.gen_df.shape[0])]
            visuals_dict['meshes_pred'] = meshes_pred
        
        if hasattr(self, 'ply_translation'):
            visuals_dict['ply_translation'] = self.ply_translation.cpu().numpy()
            visuals_dict['part_translation'] = self.part_translation.cpu().numpy()
            mesh_extents = torch.zeros(0, 3)
            for mesh in meshes:
                mesh_extents = torch.cat([mesh_extents, torch.tensor(mesh.extents).unsqueeze(0)], dim=0)
            # visuals_dict['part_scale'] = (torch.max(self.part_extent, dim=1)[0] / torch.max(mesh_extents, dim=1)[0]).cpu().numpy()
            visuals_dict['part_scale'] = (torch.max(self.part_extent, dim=1)[0] / (4 / 2.2)).cpu().numpy()

        return visuals_dict

    def save(self, label, global_step, save_opt=False):

        state_dict = {
            # 'vqvae': self.vqvae.state_dict(),
            'df': self.df.state_dict(),
            'global_step': global_step,
        }

        for i, optimizer in enumerate(self.optimizers):
            state_dict[f'opt{i}'] = optimizer.state_dict()
        for i, scheduler in enumerate(self.schedulers):
            state_dict[f'sch{i}'] = scheduler.state_dict()
           
        save_filename = 'df_%s.pth' % (label)
        save_path = os.path.join(self.opt.ckpt_dir, save_filename)

        torch.save(state_dict, save_path)

    def load_ckpt(self, ckpt, load_opt=False):
        map_fn = lambda storage, loc: storage
        if type(ckpt) == str:
            state_dict = torch.load(ckpt, map_location=map_fn)
        else:
            state_dict = ckpt

        # self.vqvae.load_state_dict(state_dict['vqvae'])
        self.df.load_state_dict(state_dict['df'])
        print(colored('[*] weight successfully load from: %s' % ckpt, 'blue'))

        # if 'opt' in state_dict:
        for i, optimizer in enumerate(self.optimizers):
            try:
                optimizer.load_state_dict(state_dict[f'opt{i}'])
            except:
                pass
        print(colored('[*] optimizer successfully restored from: %s' % ckpt, 'blue'))
        iter_passed = state_dict['global_step']
        
        if 'sch' in state_dict:
            for i, scheduler in enumerate(self.schedulers):
                try:
                    scheduler.load_state_dict(state_dict[f'sch{i}'])
                except:
                    for _ in range(state_dict['global_step']):
                        scheduler.step()
            print(colored('[*] scheduler successfully restored from: %s' % ckpt, 'blue'))

        return iter_passed

    def set_planner(self, planner):
        if not self.isTrain:
            self.planner = planner
            if self.planner is not None:
                print(colored('[*] planner type: %s' % planner.__class__.__name__,
                            'red'))
            else:
                print(colored('[*] planner type: None', 'red'))
        else:
            raise NotImplementedError('planner setter is only for inference')

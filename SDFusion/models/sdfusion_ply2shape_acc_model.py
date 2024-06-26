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
from models.loss_utils import get_physical_loss

from diffusers import DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

from accelerate import Accelerator

from utils.util import AverageMeter

# rendering
from utils.util_3d import init_mesh_renderer, render_sdf

class SDFusionModelPly2ShapeAcc(BaseModel):
    def name(self):
        return 'SDFusion-Model-PointCloud-to-Shape-Accelerate'

    def __init__(self, opt, accelerator: Accelerator):
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
        self.df = DiffusionUNet(unet_params, vq_conf=vq_conf, conditioning_key=df_conf.model.params.conditioning_key)
        self.df.to(self.device)
        self.parameterization = "eps"
        self.guidance_scale = opt.uc_scale

        # init vqvae
        self.vqvae = load_vqvae(vq_conf, vq_ckpt=opt.vq_ckpt, opt=opt).to(self.device)

        # init U-Net conditional model
        self.cond_model = PointNet2(hidden_dim=df_conf.unet.params.context_dim).to(self.device)

        # convert to sync-bn
        self.cond_model = nn.SyncBatchNorm.convert_sync_batchnorm(self.cond_model)
        self.cond_model.requires_grad_(True)

        load_result = self.cond_model.load_state_dict(torch.load(opt.cond_ckpt)['model_state_dict'], strict=False)
        print(load_result)
        print(colored('[*] conditional model successfully loaded', 'blue'))
        
        self.uncond_prob = df_conf.model.params.uncond_prob

        if self.isTrain:
            # initialize optimizers
            self.optimizer1 = optim.AdamW([p for p in self.df.parameters() if p.requires_grad == True], lr=opt.lr)
            self.optimizer2 = optim.AdamW([p for p in self.cond_model.parameters() if p.requires_grad == True], lr=opt.lr)

            lr_lambda1 = lambda it: 0.5 * (1 + np.cos(np.pi * it / opt.total_iters))
            self.scheduler1 = optim.lr_scheduler.LambdaLR(self.optimizer1, lr_lambda1)
            
            freeze_iters = opt.freeze_iters
            lr_lambda2 = lambda it: 0.5 * (1 + np.cos(np.pi * it / opt.total_iters)) if it > freeze_iters else 0
            self.scheduler2 = optim.lr_scheduler.LambdaLR(self.optimizer2, lr_lambda2)

            self.optimizers = [self.optimizer1, self.optimizer2]
            self.schedulers = [self.scheduler1, self.scheduler2]

            if opt.continue_train:
                self.start_iter = self.load_ckpt(ckpt=os.path.join(opt.ckpt_dir, f'df_steps-{opt.load_iter}.pth'))
            else:
                self.start_iter = 0

            self.optimizer1, self.optimizer2 = accelerator.prepare(self.optimizer1, self.optimizer2)
            self.scheduler1, self.scheduler2 = accelerator.prepare(self.scheduler1, self.scheduler2)

            self.print_networks(verbose=False)

        # prepare accelerate
        self.df, self.vqvae, self.cond_model = accelerator.prepare(self.df, self.vqvae, self.cond_model)

        # noise scheduler
        self.noise_scheduler = DDIMScheduler()

        ######## END: Define Networks ########

        # setup renderer
        dist, elev, azim = 1.0, 20, 120
        self.renderer = init_mesh_renderer(image_size=256, dist=dist, elev=elev, azim=azim, device=self.device)

        self.ddim_steps = self.df_conf.model.params.ddim_steps
        cprint(f'[*] setting ddim_steps={self.ddim_steps}', 'blue')
        
        self.loss_meter = AverageMeter()
        self.loss_meter.reset()
        self.loss_meter_epoch = AverageMeter()
        self.loss_meter_epoch.reset()
    
    def set_input(self, input=None, max_sample=None):
        
        self.x = input['sdf'].to(self.device)
        self.ply = input['ply'].to(self.device)
        self.paths = input['path']
        
        # transformation info for calculating collision loss
        # "ply_points + ply_translation" aligns with "part * part_extent + part_translation"
        if 'ply_translation' in input:
            self.ply_translation = input['ply_translation'].to(self.device)
            self.ply_rotation = input['ply_rotation'].to(self.device)
            self.ply_scale = input['ply_scale'].to(self.device)
            self.part_translation = input['part_translation'].to(self.device)
            self.part_extent = input['part_extent'].to(self.device)

    def switch_train(self):
        self.df.train()
        self.cond_model.train()
        self.vqvae.eval()

    def switch_eval(self):
        self.df.eval()
        self.cond_model.eval()
        self.vqvae.eval()

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
        self.start_iter += 1

        B = self.x.shape[0]
        c = self.cond_model(self.ply).unsqueeze(1) # (B, 1, context_dim)
        uc = self.cond_model(uncond=True).unsqueeze(0).unsqueeze(0).repeat(B, 1, 1) # (B, 1, context_dim), unconditional condition
        # uc = torch.zeros_like(c, device=self.device)
        # uc = self.cond_model(torch.zeros([B, 3, self.df_conf.ply.max_points]).to(self.device)).unsqueeze(1)
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
                  infer_all=False, max_sample=16):

        self.switch_eval()

        if not infer_all:
            self.set_input(data, max_sample=max_sample)
        else:
            self.set_input(data)
        
        if ddim_steps is None:
            ddim_steps = self.ddim_steps
            
        B = self.x.shape[0]
        shape = self.z_shape
        c = self.cond_model(self.ply).unsqueeze(1) # (B, context_dim), point cloud condition
        uc = self.cond_model(uncond=True).unsqueeze(0).unsqueeze(0).repeat(B, 1, 1)
        # uc = torch.zeros_like(c, device=self.device)
        # uc = self.cond_model(torch.zeros([B, 3, self.df_conf.ply.max_points]).to(self.device)).unsqueeze(1) # (B, context_dim), unconditional condition
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
            latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

        # decode z
        self.gen_df = self.vqvae.module.decode_no_quant(latents).detach()

    @torch.no_grad()
    def eval_metrics(self, dataloader, thres=0.0, global_step=0):

        ret = OrderedDict([
            ('loss', self.loss_meter_epoch.avg),
        ])
        self.loss_meter_epoch.reset()

        return ret

    def backward(self): # not used
        raise NotImplementedError('backward() is not used in this model')

    def optimize_parameters(self, total_steps):
        # self.set_requires_grad([self.df], requires_grad=True)

        loss = self.forward()
        avg_loss = self.accelerator.gather(loss).mean()
        self.loss_meter.update(avg_loss, self.opt.batch_size * self.accelerator.num_processes)
        self.loss_meter_epoch.update(avg_loss, self.opt.batch_size * self.accelerator.num_processes)
        self.accelerator.backward(loss)
        
        # clip grad norm
        torch.nn.utils.clip_grad_norm_(self.df.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.cond_model.parameters(), 0.1)
        
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
        # print learning rate
        for i, scheduler in enumerate(self.schedulers):
            print(f'learning rate at iter {self.start_iter}: {scheduler.get_last_lr()}', flush=True)

        return ret

    def get_current_visuals(self):

        spc = (2./self.shape_res, 2./self.shape_res, 2./self.shape_res)
        meshes = [sdf_to_mesh_trimesh(self.gen_df[i][0], spacing=spc) for i in range(self.gen_df.shape[0])]
        visuals_dict = {
            "meshes": meshes,
            "paths": self.paths,
            "points": self.ply.detach().cpu().numpy(), # (B, 3, N)
        }

        if self.opt.isTrain:
            self.img_gt = render_sdf(self.renderer, self.x).detach()
            self.img_gen_df = render_sdf(self.renderer, self.gen_df).detach()
            vis_tensor_names = [
                'img_gt',
                'img_gen_df',
            ]
            vis_ims = self.tnsrs2ims(vis_tensor_names)
            visuals = zip(vis_tensor_names, vis_ims)
            visuals_dict['img'] = OrderedDict(visuals)
        
        if hasattr(self, 'ply_translation'):
            visuals_dict['ply_translation'] = self.ply_translation.cpu().numpy()
            visuals_dict['ply_rotation'] = self.ply_rotation.cpu().numpy()
            visuals_dict['part_translation'] = self.part_translation.cpu().numpy()

            visuals_dict['part_scale'] = np.zeros([len(meshes)], dtype=np.float32)
            for i, mesh in enumerate(meshes):
                visuals_dict['part_scale'][i] = torch.max(self.part_extent[i]).item() / np.max(mesh.extents) if self.opt.scale_mode == 'max_extent' else \
                    (torch.prod(self.part_extent[i]).item() / np.prod(mesh.extents)) ** (1/3)

        return visuals_dict

    def save(self, label, global_step, save_opt=False):

        state_dict = {
            'df': self.df.module.state_dict(),
            'cond_model': self.cond_model.module.state_dict(),
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

        self.df.load_state_dict(state_dict['df'])
        print(colored('[*] weight successfully load from: %s' % ckpt, 'blue'))

        try:
            self.cond_model.load_state_dict(state_dict['cond_model'])
            print(colored('[*] cond model weight successfully load from: %s' % ckpt, 'blue'))
        except:
            print(colored('[*] cond model weight not loaded', 'red'))

        iter_passed = state_dict['global_step']
        if load_opt:
            for i, optimizer in enumerate(self.optimizers):
                optimizer.load_state_dict(state_dict[f'opt{i}'])
            for i, scheduler in enumerate(self.schedulers):
                scheduler.load_state_dict(state_dict[f'sch{i}'])
            print(colored('[*] optimizer successfully load from: %s' % ckpt, 'blue'))
        else:
            print(colored('[*] optimizer not loaded from: %s' % ckpt, 'blue'))
            for _ in range(iter_passed):
                for scheduler in self.schedulers:
                    scheduler.step()     

        return iter_passed


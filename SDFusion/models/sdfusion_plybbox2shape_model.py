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
from models.networks.bbox_networks.bbox_model import BBoxModel
from models.loss_utils import get_physical_loss

from diffusers import DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

from utils.util import AverageMeter, Logger

# rendering
from utils.util_3d import init_mesh_renderer, render_sdf, sdf_to_mesh_trimesh

class SDFusionModelPlyBBox2Shape(BaseModel):
    def name(self):
        return 'SDFusion-Model-PointCloud-BBox-to-Shape'

    def __init__(self, opt):
        super().__init__(opt)
        self.isTrain = opt.isTrain
        self.model_name = self.name()
        self.device = 'cuda'
        assert self.opt.ply_bbox_cond

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
        self.guidance_scale_ply = opt.uc_ply_scale
        self.guidance_scale_bbox = opt.uc_bbox_scale
        self.guidance_scale = opt.uc_scale

        # init vqvae
        self.vqvae = load_vqvae(vq_conf, vq_ckpt=opt.vq_ckpt, opt=opt).to(self.device)

        # init U-Net conditional model
        self.ply_cond_model = PointNet2(hidden_dim=df_conf.unet.params.context_dim // 2).to(self.device)
        self.ply_cond_model.requires_grad_(True)
        load_result = self.ply_cond_model.load_state_dict(torch.load(opt.cond_ckpt)['model_state_dict'], strict=False)
        print(load_result)
        print(colored('[*] conditional model successfully loaded', 'blue'))

        self.bbox_cond_model = BBoxModel(output_dim=df_conf.unet.params.context_dim // 2).to(self.device)
        self.bbox_cond_model.requires_grad_(True)

        self.uncond_prob = df_conf.model.params.uncond_prob

        ######## END: Define Networks ########

        if self.isTrain:

            self.optimizer1 = optim.AdamW([p for p in self.df.parameters() if p.requires_grad == True] + \
                            [p for p in self.ply_cond_model.parameters() if p.requires_grad == True] + \
                            [p for p in self.bbox_cond_model.parameters() if p.requires_grad == True], lr=opt.lr)

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
        dist, elev, azim = 1.0, 20, 120
        self.renderer = init_mesh_renderer(image_size=256, dist=dist, elev=elev, azim=azim, device=self.device)

        self.ddim_steps = self.df_conf.model.params.ddim_steps
        cprint(f'[*] setting ddim_steps={self.ddim_steps}', 'blue')
        self.loss_meter = AverageMeter()
        self.loss_meter.reset()
        self.loss_meter_epoch = AverageMeter()
        self.loss_meter_epoch.reset()

        if not self.opt.isTrain:
            self.collision_loss_meter = AverageMeter()
            self.collision_loss_meter.reset()
            self.contact_loss_meter = AverageMeter()
            self.contact_loss_meter.reset()
            self.diversity_index = 0 
            self.loss_tracker = []
            self.loss_tracker_pred = []

        self.logger = Logger(os.path.join(self.opt.img_dir, 'log.txt'))
    
    def set_input(self, input=None):

        self.x = input['sdf'].to(self.device)
        self.ply = input['ply'].to(self.device)
        self.paths = input['path']

        # transformation info for calculating collision loss
        # "ply_points + ply_translation" aligns with "part * part_extent + part_translation"
        if 'ply_translation' in input:
            self.ply_translation = input['ply_translation'].to(self.device)
            self.ply_rotation = input['ply_rotation'].to(self.device)
            self.part_translation = input['part_translation'].to(self.device)
            self.part_extent = input['part_extent'].to(self.device)

        if 'move_axis' in input:
            self.move_axis = input['move_axis'].to(self.device) # (3,)
            self.move_origin = input['move_origin'].to(self.device) # (3,)
            self.move_limit = input['move_limit'].to(self.device) # (2,) range of motion
        else:
            self.move_axis = self.move_limit = self.move_origin = [None] * self.x.shape[0]

        if 'ply_rotation_pred' in input:
            self.part_translation_pred = input['part_translation_pred'].to(self.device)
            self.part_extent_pred = input['part_extent_pred'].to(self.device)
            self.ply_rotation_pred = input['ply_rotation_pred'].to(self.device)
        else:
            self.part_translation_pred = self.part_translation
            self.part_extent_pred = self.part_extent
            self.ply_rotation_pred = self.ply_rotation

    def switch_train(self):
        self.df.train()
        self.ply_cond_model.train()
        self.bbox_cond_model.train()
        self.vqvae.eval()

    def switch_eval(self):
        self.df.eval()
        self.ply_cond_model.eval()
        self.bbox_cond_model.eval()
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

        c_ply = self.ply_cond_model(self.ply).unsqueeze(1) # (B, 1, ply_dim)
        uc_ply = self.ply_cond_model(uncond=True).unsqueeze(0).unsqueeze(0).repeat(B, 1, 1) # (B, 1, ply_dim), unconditional condition
        c_ply = torch.where(torch.rand(B, 1, device=self.device) < self.uncond_prob, uc_ply, c_ply)

        c_bbox = self.bbox_cond_model(self.part_extent).unsqueeze(1) # (B, 1, bbox_dim)
        uc_bbox = self.bbox_cond_model(uncond=True).unsqueeze(0).unsqueeze(0).repeat(B, 1, 1) # (B, 1, bbox_dim), unconditional condition
        c_bbox = torch.where(torch.rand(B, 1, device=self.device) < self.uncond_prob, uc_bbox, c_bbox)

        c_full = torch.cat([c_ply, c_bbox], dim=-1) # (B, 1, ply_dim + bbox_dim = context_dim)

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
        model_output = self.apply_model(z_noisy, timesteps, cond=c_full)

        if self.parameterization == "x0":
            target = z
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        loss = self.get_loss(model_output, target).mean()
        return loss

    @torch.no_grad()
    def inference(self, data, ddim_steps=None, ddim_eta=0., print_collision_loss=False,
                  use_cut_bbox=False, cut_bbox_limit=[0.5, 0.75]):

        self.switch_eval()
        self.set_input(data)
        
        if ddim_steps is None:
            ddim_steps = self.ddim_steps
        else:
            self.ddim_steps = ddim_steps
            
        B = self.x.shape[0]
        shape = self.z_shape
        
        c_ply = self.ply_cond_model(self.ply).unsqueeze(1) # (B, 1, ply_dim), point cloud condition
        uc_ply = self.ply_cond_model(uncond=True).unsqueeze(0).unsqueeze(0).repeat(B, 1, 1) # (B, 1, ply_dim), unconditional condition
        c_bbox = self.bbox_cond_model(self.part_extent_pred).unsqueeze(1) if hasattr(self, 'part_extent_pred') else \
                    self.bbox_cond_model(self.part_extent).unsqueeze(1) # (B, 1, bbox_dim), bbox condition
        uc_bbox = self.bbox_cond_model(uncond=True).unsqueeze(0).unsqueeze(0).repeat(B, 1, 1) # (B, 1, bbox_dim), unconditional condition
        
        # c_ply_uc_bbox = torch.cat([c_ply, uc_bbox], dim=-1)
        # uc_ply_c_bbox = torch.cat([uc_ply, c_bbox], dim=-1)
        # c_ply_c_bbox = torch.cat([c_ply, c_bbox], dim=-1)
        # uc_ply_uc_bbox = torch.cat([uc_ply, uc_bbox], dim=-1)
        # c_full = torch.cat([uc_ply_uc_bbox, c_ply_uc_bbox, uc_ply_c_bbox, c_ply_c_bbox])

        c_ply_c_bbox = torch.cat([c_ply, c_bbox], dim=-1)
        uc_ply_uc_bbox = torch.cat([uc_ply, uc_bbox], dim=-1)
        c_full = torch.cat([uc_ply_uc_bbox, c_ply_c_bbox])

        latents = torch.randn((B, *shape), device=self.device)
        latents = latents * self.noise_scheduler.init_noise_sigma

        self.noise_scheduler.set_timesteps(ddim_steps)

        # w/ condition
        for t in tqdm(self.noise_scheduler.timesteps):
            
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, timestep=t)

            # predict the noise residual
            timesteps = torch.full((B*2,), t, device=self.device, dtype=torch.int64)
            noise_pred = self.apply_model(latent_model_input, timesteps, c_full)

            # perform guidance
            # noise_pred_uncond, noise_pred_uc_bbox, noise_pred_uc_ply, noise_pred_cond = noise_pred.chunk(4)
            # noise_pred = noise_pred_uncond + self.guidance_scale_ply * (noise_pred_cond - noise_pred_uc_ply) + \
            #                 self.guidance_scale_bbox * (noise_pred_cond - noise_pred_uc_bbox)

            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

        # decode z
        self.gen_df = self.vqvae.decode_no_quant(latents).detach()

        # calculate physical loss
        for i in range(B):
            collision_loss, contact_loss = get_physical_loss(self.gen_df[i:i+1], self.ply[i:i+1], 
                                                self.ply_translation[i:i+1], self.ply_rotation[i:i+1],
                                                self.part_extent_pred[i:i+1] if self.opt.haoran else self.part_extent[i:i+1],
                                                self.part_translation[i:i+1],
                                                move_limit=self.move_limit[i], 
                                                move_axis=self.move_axis[i],
                                                move_origin=self.move_origin[i],
                                                move_type=self.opt.mobility_type,
                                                move_samples=self.opt.mobility_sample_count, res=self.shape_res,
                                                scale_mode=self.opt.scale_mode,
                                                margin=self.opt.loss_margin,
                                                loss_collision_weight=self.opt.loss_collision_weight,
                                                loss_contact_weight=self.opt.loss_contact_weight,
                                                use_bbox=False, linspace=True)
            instance_name = self.paths[i].split('/')[-1].split('.')[0]
            if not self.opt.test_diversity:
                self.logger.log(f'part {instance_name}, collision loss {collision_loss.item():.4f}, contact loss {contact_loss.item():.4f}')
                self.collision_loss_meter.update(collision_loss.item())
                self.contact_loss_meter.update(contact_loss.item())
            else:
                if self.opt.haoran:
                    collision_loss_pred, contact_loss_pred = get_physical_loss(self.gen_df[i:i+1], self.ply[i:i+1], 
                                                        self.ply_translation[i:i+1], self.ply_rotation_pred[i:i+1],
                                                        self.part_extent_pred[i:i+1], self.part_translation_pred[i:i+1],
                                                        move_limit=self.move_limit[i], 
                                                        move_axis=self.move_axis[i],
                                                        move_origin=self.move_origin[i],
                                                        move_type=self.opt.mobility_type,
                                                        move_samples=self.opt.mobility_sample_count, res=self.shape_res,
                                                        scale_mode=self.opt.scale_mode,
                                                        margin=self.opt.loss_margin,
                                                        loss_collision_weight=self.opt.loss_collision_weight,
                                                        loss_contact_weight=self.opt.loss_contact_weight,
                                                        use_bbox=False, linspace=True)
                else:
                    collision_loss_pred, contact_loss_pred = collision_loss, contact_loss
                self.loss_tracker.append((collision_loss.item(), contact_loss.item()))
                self.loss_tracker_pred.append((collision_loss_pred.item(), contact_loss_pred.item()))
                self.logger.log(f'part {instance_name} instance {self.diversity_index}, collision loss {collision_loss_pred:.4f}, contact loss {contact_loss_pred:.4f}')
                self.diversity_index += 1
                if self.diversity_index % self.opt.diversity_count == 0:
                    # find the best prediction
                    best_idx = np.argmin([loss[0] + loss[1] for loss in self.loss_tracker_pred])
                    best_loss = self.loss_tracker[best_idx]
                    self.logger.log(f'part {instance_name} best {best_idx}, collision loss {best_loss[0]:.4f}, contact loss {best_loss[1]:.4f}')
                    self.diversity_index = 0
                    self.loss_tracker, self.loss_tracker_pred = [], []
                    self.collision_loss_meter.update(best_loss[0])
                    self.contact_loss_meter.update(best_loss[1])

    def guided_inference(self, data, ddim_steps=None, ddim_eta=0.):
        
        self.switch_eval()
        self.set_input(data)

        if ddim_steps is None:
            ddim_steps = self.ddim_steps
        else:
            self.ddim_steps = ddim_steps

        B = self.x.shape[0] # B == 1, only support batch size 1 for now
        shape = self.z_shape

        c_ply = self.ply_cond_model(self.ply).unsqueeze(1) # (B, 1, ply_dim), point cloud condition
        uc_ply = self.ply_cond_model(uncond=True).unsqueeze(0).unsqueeze(0).repeat(B, 1, 1) # (B, 1, ply_dim), unconditional condition
        c_bbox = self.bbox_cond_model(self.part_extent_pred).unsqueeze(1) if hasattr(self, 'part_extent_pred') else \
                    self.bbox_cond_model(self.part_extent).unsqueeze(1) # (B, 1, bbox_dim), bbox condition
        uc_bbox = self.bbox_cond_model(uncond=True).unsqueeze(0).unsqueeze(0).repeat(B, 1, 1) # (B, 1, bbox_dim), unconditional condition
        
        c_ply_c_bbox = torch.cat([c_ply, c_bbox], dim=-1)
        uc_ply_uc_bbox = torch.cat([uc_ply, uc_bbox], dim=-1)
        c_full = torch.cat([uc_ply_uc_bbox, c_ply_c_bbox])

        latents = torch.randn((B, *shape), device=self.device)
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

            # compute the previous noisy sample x_t -> x_t-1
            if i > ddim_steps // 2:
                with torch.enable_grad():
                    latents_grad = latents.detach().requires_grad_(True)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    pred_x0 = self.noise_scheduler.step(noise_pred, t, latents_grad, eta=ddim_eta).pred_original_sample

                    pred_x0_sdf = self.vqvae.decode_no_quant(pred_x0)
                    # setattr(self, f'pred_sdf_x0_{i}', pred_x0_sdf.detach())

                    collision_loss, contact_loss = get_physical_loss(pred_x0_sdf, self.ply, 
                                                        self.ply_translation, self.ply_rotation_pred,
                                                        self.part_extent_pred, self.part_translation_pred,
                                                        move_limit=self.move_limit[0], 
                                                        move_axis=self.move_axis[0],
                                                        move_origin=self.move_origin[0],
                                                        move_type=self.opt.mobility_type,
                                                        move_samples=self.opt.mobility_sample_count, res=self.shape_res,
                                                        scale_mode=self.opt.scale_mode,
                                                        margin=self.opt.loss_margin,
                                                        loss_collision_weight=self.opt.loss_collision_weight,
                                                        loss_contact_weight=self.opt.loss_contact_weight,
                                                        use_bbox=False, linspace=True)

                    # print(f'collision {collision_loss.item():.4f}, contact {contact_loss.item():.4f}')
                    
                    grad = torch.autograd.grad(collision_loss + contact_loss, latents_grad)[0] # (B, *shape)
                    # print(grad.sum().item())
                    grad = grad / (grad.norm() + 1e-8) # clip grad norm
                    noise_pred = noise_pred_uncond + (1 - self.noise_scheduler.alphas_cumprod[t]) ** 0.5 * grad + \
                                    self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

            else:
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            latents = self.noise_scheduler.step(noise_pred, t, latents, eta=ddim_eta).prev_sample

        # decode z
        self.gen_df = self.vqvae.decode_no_quant(latents).detach()

        with torch.no_grad():
            collision_loss, contact_loss = get_physical_loss(self.gen_df, self.ply, 
                                                self.ply_translation, self.ply_rotation,
                                                self.part_extent_pred if self.opt.haoran else self.part_extent,
                                                self.part_translation,
                                                move_limit=self.move_limit[0], 
                                                move_axis=self.move_axis[0],
                                                move_origin=self.move_origin[0],
                                                move_type=self.opt.mobility_type,
                                                move_samples=self.opt.mobility_sample_count, res=self.shape_res,
                                                scale_mode=self.opt.scale_mode,
                                                margin=self.opt.loss_margin,
                                                loss_collision_weight=self.opt.loss_collision_weight,
                                                loss_contact_weight=self.opt.loss_contact_weight,
                                                use_bbox=False, linspace=True)
            instance_name = self.paths[0].split('/')[-1].split('.')[0]
            if not self.opt.test_diversity:
                self.logger.log(f'part {instance_name}, collision loss {collision_loss.item():.4f}, contact loss {contact_loss.item():.4f}')
                self.collision_loss_meter.update(collision_loss.item())
                self.contact_loss_meter.update(contact_loss.item())
            else:
                if self.opt.haoran:
                    collision_loss_pred, contact_loss_pred = get_physical_loss(self.gen_df, self.ply, 
                                                        self.ply_translation, self.ply_rotation_pred,
                                                        self.part_extent_pred, self.part_translation_pred,
                                                        move_limit=self.move_limit[0], 
                                                        move_axis=self.move_axis[0],
                                                        move_origin=self.move_origin[0],
                                                        move_type=self.opt.mobility_type,
                                                        move_samples=self.opt.mobility_sample_count, res=self.shape_res,
                                                        scale_mode=self.opt.scale_mode,
                                                        margin=self.opt.loss_margin,
                                                        loss_collision_weight=self.opt.loss_collision_weight,
                                                        loss_contact_weight=self.opt.loss_contact_weight,
                                                        use_bbox=False, linspace=True)
                else:
                    collision_loss_pred, contact_loss_pred = collision_loss, contact_loss
                self.loss_tracker.append((collision_loss.item(), contact_loss.item()))
                self.loss_tracker_pred.append((collision_loss_pred.item(), contact_loss_pred.item()))
                self.logger.log(f'part {instance_name} instance {self.diversity_index}, collision loss {collision_loss_pred:.4f}, contact loss {contact_loss_pred:.4f}')
                self.diversity_index += 1
                if self.diversity_index % self.opt.diversity_count == 0:
                    best_idx = np.argmin([loss[0] + loss[1] for loss in self.loss_tracker_pred])
                    best_loss = self.loss_tracker[best_idx]
                    self.logger.log(f'part {instance_name} best {best_idx}, collision loss {best_loss[0]:.4f}, contact loss {best_loss[1]:.4f}')
                    self.diversity_index = 0
                    self.loss_tracker, self.loss_tracker_pred = [], []
                    self.collision_loss_meter.update(best_loss[0])
                    self.contact_loss_meter.update(best_loss[1])

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
        self.loss_meter.update(avg_loss, self.opt.batch_size)
        self.loss_meter_epoch.update(avg_loss, self.opt.batch_size)
        loss.backward()
        
        # clip grad norm
        torch.nn.utils.clip_grad_norm_(self.df.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.ply_cond_modell.parameters(), 1.0)
        
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

        if hasattr(self, f'pred_sdf_x0_{self.ddim_steps-1}'):
            meshes_pred = [sdf_to_mesh_trimesh(getattr(self, f'pred_sdf_x0_{i}')[0][0], spacing=spc) for i in range(self.ddim_steps)]
            visuals_dict['meshes_pred'] = meshes_pred
        
        if hasattr(self, 'ply_translation'):
            visuals_dict['ply_translation'] = self.ply_translation.cpu().numpy()
            visuals_dict['ply_rotation'] = self.ply_rotation.cpu().numpy()
            visuals_dict['part_translation'] = self.part_translation.cpu().numpy()

            visuals_dict['part_scale'] = np.zeros([len(meshes)], dtype=np.float32)
            for i, mesh in enumerate(meshes):
                if self.opt.haoran:
                    visuals_dict['part_scale'][i] = torch.max(self.part_extent_pred[i]).item() / np.max(mesh.extents) if self.opt.scale_mode == 'max_extent' else \
                        (torch.prod(self.part_extent_pred[i]).item() / np.prod(mesh.extents)) ** (1/3)
                else:
                    visuals_dict['part_scale'][i] = torch.max(self.part_extent[i]).item() / np.max(mesh.extents) if self.opt.scale_mode == 'max_extent' else \
                        (torch.prod(self.part_extent[i]).item() / np.prod(mesh.extents)) ** (1/3)

        return visuals_dict

    def save(self, label, global_step, save_opt=False):

        state_dict = {
            'df': self.df.module.state_dict(),
            'cond_model_ply': self.ply_cond_model.state_dict(),
            'cond_model_bbox': self.bbox_cond_model.state_dict(),
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

        self.ply_cond_model.load_state_dict(state_dict['cond_model_ply'])
        print(colored('[*] pcd conditional model successfully restored from: %s' % ckpt, 'blue'))

        self.bbox_cond_model.load_state_dict(state_dict['cond_model_bbox'])
        print(colored('[*] bbox conditional model successfully restored from: %s' % ckpt, 'blue'))
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

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

from planners.base_model import create_planner

# rendering
from utils.util_3d import init_mesh_renderer, render_sdf

class SDFusionModelPly2ShapeAcc(BaseModel):
    def name(self):
        return 'SDFusion-Model-PointCloud-to-Shape-Refinement-Accelerate'

    def __init__(self, opt, sdf, accelerator: Accelerator):
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
        # self.init_diffusion_params(scale=1, opt=opt)

        # init vqvae
        self.vqvae = load_vqvae(vq_conf, vq_ckpt=opt.vq_ckpt, opt=opt).to(self.device)
        self.vqvae.requires_grad_(False)

        # init U-Net conditional model
        self.cond_model = PointNet2(hidden_dim=df_conf.unet.params.context_dim).to(self.device)
        # convert to sync-bn
        self.cond_model = nn.SyncBatchNorm.convert_sync_batchnorm(self.cond_model)
        self.cond_model.requires_grad_(False)

        if self.isTrain:

            self.optimizer1 = optim.AdamW([p for p in self.df.parameters() if p.requires_grad == True] + \
                            [p for p in self.cond_model.parameters() if p.requires_grad == True], lr=opt.lr)

            lr_lambda1 = lambda it: 0.5 * (1 + np.cos(np.pi * it / opt.total_iters))
            self.scheduler1 = optim.lr_scheduler.LambdaLR(self.optimizer1, lr_lambda1)

            self.optimizers = [self.optimizer1]
            self.schedulers = [self.scheduler1]

            self.print_networks(verbose=False)

        # prepare accelerate
        self.df, self.vqvae, self.cond_model = accelerator.prepare(self.df, self.vqvae, self.cond_model)

        # noise scheduler
        self.noise_scheduler = DDIMScheduler()

        ######## END: Define Networks ########

        self.sdf = nn.Parameter(sdf, requires_grad=True)
        self.batch_size = opt.batch_size

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

        self.loss_names = ['sds', 'collision', 'total']
        self.loss_meter_dict = {name: AverageMeter() for name in self.loss_names}
        self.loss_meter_epoch_dict = {name: AverageMeter() for name in self.loss_names}

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
    
    def get_sdf_refinement_loss(self, latents: torch.Tensor, ply: torch.Tensor):
        '''
        latent: the latent representation of the input sdf
        calculate the sdf refinement loss
        including: sds loss and collision loss
        return the total loss
        '''

        self.switch_eval()
        loss_dict ={}
        
        # 1. sds loss
        B = latents.shape[0] # latents: (B, latent_dim, res, res, res)
        c = self.cond_model(ply).unsqueeze(1) # (B, 1, context_dim)
        uc = self.cond_model(uncond=True).unsqueeze(0).unsqueeze(0).repeat(B, 1, 1)
        c_full = torch.cat([uc, c])

        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (B*2,), device=latents.device,
            dtype=torch.int64
        )

        noise = torch.randn(latents.shape, device=latents.device)
        # Add noise to the clean images according to the noise magnitude at each timestep
        latent_noisy = self.noise_scheduler.add_noise(latents, noise, timesteps)
        latent_model_input = torch.cat([latent_noisy] * 2)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = self.apply_model(latent_model_input, timesteps, c_full)

        # perform guidance
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

        weight = self.noise_scheduler.alphas_cumprod[timesteps].view(B, 1, 1, 1, 1)

        grad = (noise_pred - noise) * weight
        target = (latents - grad).detach()

        loss_sds = F.mse_loss(latents, target, reduction="mean")
        
        loss_dict['sds'] = loss_sds
    

    def forward(self):

        self.switch_eval()
        return self.get_sdf_refinement_loss()

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
        torch.nn.utils.clip_grad_norm_(self.sdf, 0.1)
        
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
            print(f'learning rate at iter {self.start_iter}: {scheduler.get_last_lr()}', flush=True)

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
        }
        return visuals_dict

    def save(self, label, global_step, save_opt=False):

        state_dict = {
            # 'vqvae': self.vqvae.module.state_dict(),
            'df': self.df.module.state_dict(),
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

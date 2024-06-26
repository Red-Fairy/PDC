import os
from collections import OrderedDict

import numpy as np
import mcubes
import omegaconf
from termcolor import colored
from einops import rearrange
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.profiler import record_function
import torch.nn.functional as F
from termcolor import colored, cprint

import torchvision.utils as vutils
import torchvision.transforms as transforms

from models.base_model import BaseModel
from models.networks.cvae_networks.network import CVAE
from models.loss_utils import VQLoss

from utils.util import AverageMeter, Logger

from utils.util_3d import init_mesh_renderer, sdf_to_mesh_trimesh

class CVAEModelPly2Shape(BaseModel):
    def name(self):
        return 'CVAE-Model-PointCloud-to-Shape'
    
    def __init__(self, opt):
        super().__init__(opt)
        self.isTrain = opt.isTrain
        self.model_name = self.name()
        self.device = 'cuda'

        # -------------------------------
        # Define Networks
        # -------------------------------

        assert opt.cvae_cfg is not None
        configs = omegaconf.OmegaConf.load(opt.cvae_cfg)
        # embed_dim = configs.model.params.embed_dim
        ddconfig = configs.model.params.ddconfig
        self.loss_weight_annealing = configs.lossconfig.loss_weight_annealing
        self.loss_recon_weight = configs.lossconfig.params.recon_weight
        self.loss_kld_weight = configs.lossconfig.params.kld_weight

        self.cvae = CVAE(ddconfig, configs.model.params.condconfig, opt.cond_ckpt)
        self.cvae.to(self.device)

        # record z_shape
        self.shape_res = ddconfig.resolution
        z_ch, n_down = ddconfig.z_channels, len(ddconfig.ch_mult)-1
        z_sp_dim = self.shape_res // (2 ** n_down)
        self.z_shape = (z_ch, z_sp_dim, z_sp_dim, z_sp_dim)
        
        ######## END: Define Networks ########

        cprint(f"[*] Model {self.model_name} initialized, isTrain:{self.isTrain}", 'green')
        if self.isTrain:

            # self.optimizer1 = optim.AdamW([p for p in self.cvae.parameters() if p.requires_grad == True] + \
            #                 [p for p in self.cond_model.parameters() if p.requires_grad == True], lr=opt.lr)
            self.optimizer1 = optim.AdamW([p for p in self.cvae.parameters() if p.requires_grad == True])

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
        
        # setup renderer
        dist, elev, azim = 1.0, 20, 120
        self.renderer = init_mesh_renderer(image_size=256, dist=dist, elev=elev, azim=azim, device='cpu')

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

        if 'bbox_mesh' in input:
            self.bbox_mesh = input['bbox_mesh'].to(self.device)

        if 'ply_rotation_pred' in input:
            self.part_translation_pred = input['part_translation_pred'].to(self.device)
            self.part_extent_pred = input['part_extent_pred'].to(self.device)
            self.ply_rotation_pred = input['ply_rotation_pred'].to(self.device)

    def switch_train(self):
        self.cvae.train()

    def switch_eval(self):
        self.cvae.eval()

    def apply_model(self, x_noisy, t, cond, return_ids=False):
        raise NotImplementedError
    
    def get_loss(self, x_target, x_recon, z_latent, expoch, mean=True):
        raise NotImplementedError


    def forward(self):
        self.switch_train()

        B = self.x.shape[0]
        
        z, mu, logvar = self.cvae.encode(self.x, self.ply, verbose=True)
        x_recon = self.cvae.decode(z, self.ply)

        loss = 0.
        ## 1. recon loss
        rec_loss = F.l1_loss(x_recon, self.x)
        loss += self.loss_recon_weight * rec_loss

        ## 2. kl divergence loss
        kl_loss = 0.5 * torch.mean(torch.exp(logvar) + mu**2 - 1.0 - logvar)
        loss += self.loss_kld_weight * kl_loss

        return loss
    
    def inference(self, data, **kwargs):
        
        self.switch_eval()
        self.set_input(data)

        B = self.x.shape[0]
        eps = torch.randn(B, *self.z_shape).to(self.device)
        self.gen_df = self.cvae.decode(eps, self.ply).detach()

    def guided_inference(self):
        raise NotImplementedError('Guided Inference not implemented for this model')

    @torch.no_grad()
    def eval_metrics(self, test_dl, global_step):
        ret = OrderedDict([
            ('loss', self.loss_meter_epoch.avg),
        ])
        self.loss_meter_epoch.reset()

        return ret
    
    def backward(self):
        raise NotImplementedError('backward() is not used in this model')
    
    def optimize_parameters(self, total_steps):
        loss = self.forward()
        avg_loss = loss.mean()
        if avg_loss > 100:
            print(colored(f"Loss too large: {avg_loss.item()} | operation: skipping", 'red'))
            return
        self.loss_meter.update(avg_loss, self.opt.batch_size)
        self.loss_meter_epoch.update(avg_loss, self.opt.batch_size)
        loss.backward()

        # clip grad norm
        torch.nn.utils.clip_grad_norm_(self.cvae.parameters(), 1.0)
        # torch.nn.utils.clip_grad_norm_(self.cond_model.parameters(), 1.0)
        
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
    
    @torch.no_grad()
    def get_current_visuals(self):
        spc = (2./self.shape_res, 2./self.shape_res, 2./self.shape_res)
        meshes = [sdf_to_mesh_trimesh(self.gen_df[i][0], spacing=spc) for i in range(self.gen_df.shape[0])]
        visuals_dict = {
            "meshes": meshes,
            "paths": self.paths,
            "points": self.ply.detach().cpu().numpy(), # (B, 3, N)
        }

        # if self.opt.isTrain:
        #     self.img_gt = render_sdf(self.renderer, self.x.to('cpu')).detach()
        #     self.img_gen_df = render_sdf(self.renderer, self.gen_df.to('cpu')).detach()
        #     vis_tensor_names = [
        #         'img_gt',
        #         'img_gen_df',
        #     ]
        #     vis_ims = self.tnsrs2ims(vis_tensor_names)
        #     visuals = zip(vis_tensor_names, vis_ims)
        #     visuals_dict['img'] = OrderedDict(visuals)
        
        if hasattr(self, 'ply_translation'):
            visuals_dict['ply_translation'] = self.ply_translation.cpu().numpy()
            visuals_dict['ply_rotation'] = self.ply_rotation.cpu().numpy()
            visuals_dict['part_translation'] = self.part_translation.cpu().numpy()

            visuals_dict['part_scale'] = np.zeros([len(meshes)], dtype=np.float32)
            for i, mesh in enumerate(meshes):
                visuals_dict['part_scale'][i] = torch.max(self.part_extent[i]).item() / np.max(mesh.extents) if self.opt.scale_mode == 'max_extent' else \
                    (torch.prod(self.part_extent[i]).item() / np.prod(mesh.extents)) ** (1/3)

        return visuals_dict

    @torch.no_grad()
    def save(self, label, global_step, save_opt=False):
        state_dict = {
            'cvae': self.cvae.state_dict(),
            # 'cond_model': self.cond_model.state_dict(),
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
            
        self.cvae.load_state_dict(state_dict['cvae'])
        print(colored('[*] weight successfully load from: %s' % ckpt, 'blue'))

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

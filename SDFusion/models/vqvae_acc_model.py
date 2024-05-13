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

import torchvision.utils as vutils
import torchvision.transforms as transforms

from models.base_model import BaseModel
from models.networks.vqvae_networks.network import VQVAE
from models.loss_utils import VQLoss

import utils.util
from utils.util_3d import init_mesh_renderer, render_sdf, sdf_to_mesh_trimesh

from accelerate import Accelerator

from utils.util import AverageMeter

class VQVAEAccModel(BaseModel):
    def name(self):
        return 'VQVAE-Model-Acc'

    def __init__(self, opt, accelerator: Accelerator):
        super().__init__(opt)

        self.isTrain = opt.isTrain
        self.model_name = self.name()
        self.device = accelerator.device
        self.accelerator = accelerator

        # -------------------------------
        # Define Networks
        # -------------------------------

        # model
        assert opt.vq_cfg is not None
        configs = omegaconf.OmegaConf.load(opt.vq_cfg)
        mparam = configs.model.params
        n_embed = mparam.n_embed
        embed_dim = mparam.embed_dim
        ddconfig = mparam.ddconfig
        self.shape_res = ddconfig.resolution

        self.vqvae = VQVAE(ddconfig, n_embed, embed_dim).to(self.device)

        if self.isTrain:
            # define loss functions
            codebook_weight = configs.lossconfig.params.codebook_weight
            self.loss_vq = VQLoss(codebook_weight=codebook_weight).to(self.device)

            # initialize optimizers
            self.optimizer = optim.AdamW(self.vqvae.parameters(), lr=opt.lr, betas=(0.9, 0.999), weight_decay=1e-4)
            lr_lambda = lambda it: 0.5 * (1 + np.cos(np.pi * it / opt.total_iters))
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

            self.optimizer, self.scheduler = accelerator.prepare(self.optimizer, self.scheduler)

            self.optimizers = [self.optimizer]
            self.schedulers = [self.scheduler]

            self.print_networks(verbose=False)

        # continue training
        if opt.continue_train:
            self.start_iter = self.load_ckpt(ckpt=os.path.join(opt.ckpt_dir, f'vqvae_steps-{opt.load_iter}.pth'))
        else:
            self.start_iter = 0

        self.vqvae = accelerator.prepare(self.vqvae)

        # setup renderer
        dist, elev, azim = 1.0, 20, 120  #! require to be check
        self.renderer = init_mesh_renderer(image_size=256, dist=dist, elev=elev, azim=azim, device=self.device)

        # for saving best ckpt
        self.best_iou = -1e12

        self.loss_names = ['total', 'codebook', 'nll', 'rec']
        self.loss_meter_dict = {name: AverageMeter() for name in self.loss_names}
        self.loss_meter_epoch_dict = {name: AverageMeter() for name in self.loss_names}

    def switch_eval(self):
        self.vqvae.eval()
        
    def switch_train(self):
        self.vqvae.train()

    def set_input(self, input):
        
        self.paths = input['path']
        self.x = input['sdf'].to(self.device)
        # self.cur_bs = self.x.shape[0] # to handle last batch 

    def forward(self):

        self.switch_train()
        self.start_iter += 1

        self.x_recon, self.qloss = self.vqvae(self.x, verbose=False)

    @torch.no_grad()
    def inference(self, data, should_render=False, verbose=False):

        self.switch_eval()
        self.set_input(data)

        with torch.no_grad():
            self.z = self.vqvae(self.x, forward_no_quant=True, encode_only=True)
            self.x_recon = self.vqvae.module.decode_no_quant(self.z).detach()

            if should_render:
                self.image = render_sdf(self.renderer, self.x)
                self.image_recon = render_sdf(self.renderer, self.x_recon)

    def test_iou(self, data, thres=0.0):
        """
            thres: threshold to consider a voxel to be free space or occupied space.
        """
        self.inference(data, should_render=False)

        x = self.x
        x_recon = self.x_recon
        iou = utils.util.iou(x, x_recon, thres)

        return iou

    def eval_metrics(self, dataloader, thres=0.0, global_step=0):

        self.switch_eval()

        iou_list = []
        with torch.no_grad():
            for ix, test_data in tqdm(enumerate(dataloader), total=len(dataloader)):

                iou = self.test_iou(test_data, thres=thres)
                iou_list.append(iou.detach())

        iou = torch.cat(iou_list)
        iou_mean, iou_std = iou.mean(), iou.std()
        
        ret = OrderedDict([
            ('iou', iou_mean.data),
            ('iou_std', iou_std.data),
        ])

        # check whether to save best epoch
        if ret['iou'] > self.best_iou:
            self.best_iou = ret['iou']
            save_name = f'epoch-best'
            self.save(save_name, global_step) # pass 0 just now

        return ret

    def backward(self):
        raise NotImplementedError 

    def optimize_parameters(self, total_steps):

        self.forward()
        total_loss, loss_dict = self.loss_vq(self.qloss, self.x, self.x_recon)

        avg_loss_dict = {k: self.accelerator.gather(v).mean() for k, v in loss_dict.items()}
        for k, v in avg_loss_dict.items():
            self.loss_meter_dict[k].update(v.item(), self.opt.batch_size)
            self.loss_meter_epoch_dict[k].update(v.item(), self.opt.batch_size)

        self.accelerator.backward(total_loss)

        # clip grad norm
        # torch.nn.utils.clip_grad_norm_(self.latent, 0.1)
        
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

    def get_current_visuals(self):

        with torch.no_grad():
            self.img = render_sdf(self.renderer, self.x)
            self.img_recon = render_sdf(self.renderer, self.x_recon)

        spc = (2./self.shape_res, 2./self.shape_res, 2./self.shape_res)
        meshes = [sdf_to_mesh_trimesh(self.z[i][0], spacing=spc) for i in range(self.z.shape[0])]

        visuals_dict = {
            "meshes": meshes,
            "paths": self.paths,
        }  

        if self.opt.isTrain:
            vis_tensor_names = [
                'img',
                'img_recon',
            ]
            vis_ims = self.tnsrs2ims(vis_tensor_names)
            visuals = zip(vis_tensor_names, vis_ims)
            visuals_dict['img'] = OrderedDict(visuals)

        return visuals_dict

    def save(self, label, global_step=0, save_opt=False):

        state_dict = {
            'vqvae': self.vqvae.module.state_dict(),
            'global_step': global_step,
        }
        
        for i, optimizer in enumerate(self.optimizers):
            state_dict[f'opt{i}'] = optimizer.state_dict()
        for i, scheduler in enumerate(self.schedulers):
            state_dict[f'sch{i}'] = scheduler.state_dict()

        save_filename = 'vqvae_%s.pth' % (label)
        save_path = os.path.join(self.opt.ckpt_dir, save_filename)

        torch.save(state_dict, save_path)

    def get_codebook_weight(self):
        ret = self.vqvae.quantize.embedding.cpu().state_dict()
        self.vqvae.quantize.embedding.cuda()
        return ret

    def load_ckpt(self, ckpt, load_opt=False):
        map_fn = lambda storage, loc: storage
        if type(ckpt) == str:
            state_dict = torch.load(ckpt, map_location=map_fn)
        else:
            state_dict = ckpt
        
        # NOTE: handle version difference...
        if 'vqvae' not in state_dict:
            self.vqvae.load_state_dict(state_dict)
        else:
            self.vqvae.load_state_dict(state_dict['vqvae'])
            
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



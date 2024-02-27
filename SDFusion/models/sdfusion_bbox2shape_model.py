# Reference: diffusion is borrowed from the LDM repo: https://github.com/CompVis/latent-diffusion
# Specifically, functions from: https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/diffusion/ddpm.py
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

# ldm util
from models.networks.diffusion_networks.ldm_diffusion_util import (
    make_beta_schedule,
    extract_into_tensor,
    noise_like,
    exists,
    default,
)
from models.networks.diffusion_networks.samplers.ddim import DDIMSampler

from planners.base_model import create_planner

# distributed 
from utils.distributed import reduce_loss_dict

# rendering
from utils.util_3d import init_mesh_renderer, render_sdf, render_bbox

class PositionalEncoding(nn.Module):
    def __init__(self, min_deg=0, max_deg=5):
        super(PositionalEncoding, self).__init__()
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.scales = torch.tensor([2 ** i for i in range(min_deg, max_deg)])

    def forward(self, x, y=None):
        x_ = x
        shape = list(x.shape[:-1]) + [-1]
        x_enc = (x[..., None, :] * self.scales[:, None].to(x.device)).reshape(shape)
        x_enc = torch.cat((x_enc, x_enc + 0.5 * torch.pi), -1)
        if y is not None:
            # IPE
            y_enc = (y[..., None, :] * self.scales[:, None].to(x.device)**2).reshape(shape)
            y_enc = torch.cat((y_enc, y_enc), -1)
            x_ret = torch.exp(-0.5 * y_enc) * torch.sin(x_enc)
            y_ret = torch.maximum(torch.zeros_like(y_enc), 0.5 * (1 - torch.exp(-2 * y_enc) * torch.cos(2 * x_enc)) - x_ret ** 2)
            
            x_ret = torch.cat([x_ret, x_], dim=-1) # N*(6*(max_deg-min_deg)+3)
            return x_ret, y_ret
        else:
            # PE
            x_ret = torch.sin(x_enc)
            x_ret = torch.cat([x_ret, x_], dim=-1) # N*(6*(max_deg-min_deg)+3)
            return x_ret

class SDFusionBbox2ShapeModel(BaseModel):
    def name(self):
        return 'SDFusionBbox2ShapeModel'

    def __init__(self, opt):
        super().__init__(opt)
        self.isTrain = opt.isTrain
        self.model_name = self.name()
        self.device = opt.device

        # self.optimizer_skip = False
        ######## START: Define Networks ########
        assert opt.df_cfg is not None
        assert opt.vq_cfg is not None

        # init df
        df_conf = OmegaConf.load(opt.df_cfg)
        vq_conf = OmegaConf.load(opt.vq_cfg)

        # record z_shape
        ddconfig = vq_conf.model.params.ddconfig
        shape_res = ddconfig.resolution
        z_ch, n_down = ddconfig.z_channels, len(ddconfig.ch_mult)-1
        z_sp_dim = shape_res // (2 ** n_down)
        self.z_shape = (z_ch, z_sp_dim, z_sp_dim, z_sp_dim)
        # init diffusion networks
        unet_params = df_conf.unet.params
        self.uc_scale = df_conf.model.params.uc_scale
        self.df = DiffusionUNet(unet_params, vq_conf=vq_conf, conditioning_key=df_conf.model.params.conditioning_key)
        self.df.to(self.device)
        self.init_diffusion_params(uc_scale=self.uc_scale, opt=opt)

        if not self.isTrain:
            self.planner = create_planner(opt)

        # sampler 
        self.ddim_sampler = DDIMSampler(self)

        # init vqvae
        self.vqvae = load_vqvae(vq_conf, vq_ckpt=opt.vq_ckpt, opt=opt)

        # init cond model
        # MLP sinusoid positional encoding
        self.pos_enc = PositionalEncoding(min_deg=0, max_deg=df_conf.model.params.bbox_nfreq)
        self.uncond_prob = df_conf.model.params.uncond_prob
        ######## END: Define Networks ########
        
        # params
        trainable_models = [self.df]
        trainable_params = []
        for m in trainable_models:
            trainable_params += [p for p in m.parameters() if p.requires_grad == True]

        if self.isTrain:
            # initialize optimizers
            self.optimizer = optim.AdamW(trainable_params, lr=opt.lr)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 1000, 0.9)

            self.optimizers = [self.optimizer]
            self.schedulers = [self.scheduler]

            self.print_networks(verbose=False)

        if opt.ckpt is not None:
            self.load_ckpt(opt.ckpt, load_opt=self.isTrain)
            
        # transforms
        self.to_tensor = transforms.ToTensor()

        # setup renderer
        if 'snet' in opt.dataset_mode:
            dist, elev, azim = 1.7, 20, 20
        elif 'pix3d' in opt.dataset_mode:
            dist, elev, azim = 1.7, 20, 20
        elif opt.dataset_mode == 'buildingnet':
            dist, elev, azim = 1.0, 20, 20
        elif opt.dataset_mode == 'gapnet':
            dist, elev, azim = 1.0, 20, 20

        self.renderer = init_mesh_renderer(image_size=256, dist=dist, elev=elev, azim=azim, device=self.device)

        # for distributed training
        if self.opt.distributed:
            self.make_distributed(opt)

            self.df_module = self.df.module
            self.vqvae_module = self.vqvae.module
            # self.cond_model_module = self.cond_model.module
        else:
            self.df_module = self.df
            self.vqvae_module = self.vqvae
            # self.cond_model_module = self.cond_model

        self.ddim_steps = 200
        if self.opt.debug == "1":
            # NOTE: for debugging purpose
            self.ddim_steps = 7
        cprint(f'[*] setting ddim_steps={self.ddim_steps}', 'blue')

    def make_distributed(self, opt):
        self.df = nn.parallel.DistributedDataParallel(
            self.df,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
        )
        self.vqvae = nn.parallel.DistributedDataParallel(
            self.vqvae,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
        )
        # self.cond_model = nn.parallel.DistributedDataParallel(
        #     self.cond_model,
        #     device_ids=[opt.local_rank],
        #     output_device=opt.local_rank,
        #     broadcast_buffers=False,
        #     find_unused_parameters=True,
        # )
    ############################ START: init diffusion params ############################
    def init_diffusion_params(self, uc_scale=1., opt=None):
        
        df_conf = OmegaConf.load(opt.df_cfg)
        df_model_params = df_conf.model.params
        
        # ref: ddpm.py, line 44 in __init__()
        self.parameterization = "eps"
        self.learn_logvar = False
        
        self.v_posterior = 0.
        self.original_elbo_weight = 0.
        self.l_simple_weight = 1.
        # ref: ddpm.py, register_schedule
        self.register_schedule(
            timesteps=df_model_params.timesteps,
            linear_start=df_model_params.linear_start,
            linear_end=df_model_params.linear_end,
        )
        
        logvar_init = 0.
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,),device=self.device)
        # for cls-free guidance
        self.uc_scale = uc_scale
        
    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                        linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.betas = to_torch(betas).to(self.device)
        self.alphas_cumprod = to_torch(alphas_cumprod).to(self.device)
        self.alphas_cumprod_prev = to_torch(alphas_cumprod_prev).to(self.device)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = to_torch(np.sqrt(alphas_cumprod)).to(self.device)
        self.sqrt_one_minus_alphas_cumprod = to_torch(np.sqrt(1. - alphas_cumprod)).to(self.device)
        self.log_one_minus_alphas_cumprod = to_torch(np.log(1. - alphas_cumprod)).to(self.device)
        self.sqrt_recip_alphas_cumprod = to_torch(np.sqrt(1. / alphas_cumprod)).to(self.device)
        self.sqrt_recipm1_alphas_cumprod = to_torch(np.sqrt(1. / alphas_cumprod - 1)).to(self.device)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = to_torch(posterior_variance).to(self.device)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = to_torch(np.log(np.maximum(posterior_variance, 1e-20))).to(self.device)
        self.posterior_mean_coef1 = to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)).to(self.device)
        self.posterior_mean_coef2 = to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)).to(self.device)

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas).to(self.device) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.lvlb_weights = lvlb_weights
        assert not torch.isnan(self.lvlb_weights).all()

    def set_input(self, input=None, gen_order=None, max_sample=None):
        
        vars_list = ['bbox']
        self.bbox = input['bbox']
        self.bbox = self.bbox[:max_sample] if max_sample is not None else self.bbox

        if 'sdf' in input:
            self.x = input['sdf']
            self.path = input['path']
            vars_list.append('x')
            self.x = self.x[:max_sample] if max_sample is not None else self.x

        self.tocuda(var_names=vars_list)

    def switch_train(self):
        self.df.train()
        # self.cond_model.train()

    def switch_eval(self):
        self.df.eval()
        self.vqvae.eval()
        # self.cond_model.eval()

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    # check: ddpm.py, line 891
    def apply_model(self, x_noisy, t, cond, return_ids=False):

        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            # key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            key = 'c_concat' if self.df_module.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}
        # import pdb; pdb.set_trace()
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

    # check: ddpm.py, line 871 forward
    # check: p_losses
    # check: q_sample, apply_model
    def p_losses(self, x_start, cond, t, noise=None):

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # predict noise (eps) or x0
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        # l2
        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3, 4])
        loss = self.l_simple_weight * loss_simple.mean()
        loss_dict.update({f'loss': loss.mean()})
    
        return x_noisy, target, loss, loss_dict


    def forward(self):
        self.switch_train()
        # import pdb; pdb.set_trace()
        # c_img = self.cond_model(self.img).float()
        # change axis
        
        # c_bbox = torch.tensor(self.bbox)[:,1,:].repeat(16,16,16,1,1).float().to(self.device).permute((3,4,0,1,2))

        # 1. encode to latent
        #    encoder, quant_conv, but do not quantize
        #    check: ldm.models.autoencoder.py, VQModelInterface's encode(self, x)
        with torch.no_grad():
            z = self.vqvae(self.x, forward_no_quant=True, encode_only=True).detach() # (B, embed_dim, latent_shape, latent_shape, latent_shape)

        c_bbox = self.pos_enc(torch.tensor(self.bbox)[:,1,:].float().to(self.device)) # (B, 6*(max_deg-min_deg)+3)
        c_bbox = c_bbox.unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1,1,*self.z_shape[1:])
        c_bbox[torch.where(torch.rand(c_bbox.shape[0]) < self.uncond_prob)] = 0.

        # 2. do diffusion's forward
        t = torch.randint(0, self.num_timesteps, (z.shape[0],), device=self.device).long()
        z_noisy, target, loss, loss_dict = self.p_losses(z, c_bbox, t)

        self.loss_dict = loss_dict

    # check: ddpm.py, log_images(). line 1317~1327
    @torch.no_grad()
    def inference(self, data, ddim_steps=None, ddim_eta=0., uc_scale=None,
                  infer_all=False, max_sample=16):

        self.switch_eval()

        if not infer_all:
            self.set_input(data, max_sample=max_sample)
        else:
            self.set_input(data)

        if ddim_steps is None:
            ddim_steps = self.ddim_steps

        if uc_scale is None:
            uc_scale = self.uc_scale

        # get noise, denoise, and decode with vqvae
        # c_bbox = torch.tensor(self.bbox)[:,1,:].repeat(16,16,16,1,1).float().to(self.device).permute((3,4,0,1,2)) #self.bbox.reshape(self.bbox.shape[0], -1)

        c_bbox = self.pos_enc(torch.tensor(self.bbox)[:,1,:].float().to(self.device)) # (B, 6*(max_deg-min_deg)+3)
        c_bbox = c_bbox.unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1,1,*self.z_shape[1:])
        # c_bbox[torch.where(torch.rand(c_bbox.shape[0]) < self.uncond_prob)] = 0.
        uc_bbox = torch.zeros_like(c_bbox).to(self.device)
        B = c_bbox.shape[0]

        shape = self.z_shape
        samples, intermediates = self.ddim_sampler.sample(S=ddim_steps,
                                                        batch_size=B,
                                                        shape=shape,
                                                        conditioning=c_bbox,
                                                        verbose=False,
                                                        unconditional_guidance_scale=uc_scale,
                                                        unconditional_conditioning=uc_bbox,
                                                        eta=ddim_eta,
                                                        quantize_x0=False)


        self.gen_df = self.vqvae_module.decode_no_quant(samples)

        self.switch_train()

    @torch.no_grad()
    def img2shape(self, image, mask, ddim_steps=None, ddim_eta=0., uc_scale=None,
                  infer_all=False, max_sample=16):
        import pdb; pdb.set_trace()
        #######################
        ### preprocess data ###
        from utils.demo_util import preprocess_image
        import torchvision.transforms as transforms
        
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.Resize((256, 256)),
        ])
        
        _, img = preprocess_image(image, mask)
        img = transforms(img)
        self.img = img.unsqueeze(0).to(self.device)
        self.uc_img = torch.zeros_like(self.img).to(self.device)
        #######################

        # real inference
        self.switch_eval()

        if ddim_steps is None:
            ddim_steps = self.ddim_steps

        if uc_scale is None:
            uc_scale = self.uc_scale

        # get noise, denoise, and decode with vqvae
        uc = self.cond_model(self.uc_img).float() # img shape
        c_img = self.cond_model(self.img).float()
        B = c_img.shape[0]
        shape = self.z_shape
        samples, intermediates = self.ddim_sampler.sample(S=ddim_steps,
                                                        batch_size=B,
                                                        shape=shape,
                                                        conditioning=c_img,
                                                        verbose=False,
                                                        unconditional_guidance_scale=uc_scale,
                                                        unconditional_conditioning=uc,
                                                        eta=ddim_eta,
                                                        quantize_x0=False)


        self.gen_df = self.vqvae_module.decode_no_quant(samples)
        return self.gen_df

    @torch.no_grad()
    def eval_metrics(self, dataloader, thres=0.0, global_step=0):
        self.switch_eval()
        
        ret = OrderedDict([
            ('dummy_metrics', 0.0),
        ])

        self.switch_train()
        return ret

    def backward(self):
        
        self.loss_dict = reduce_loss_dict(self.loss_dict)
        self.loss = self.loss_dict['loss']

        self.loss.backward()

    def optimize_parameters(self, total_steps):
        self.set_requires_grad([self.df], requires_grad=True)
        # self.set_requires_grad([self.cond_model], requires_grad=True)
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        
        ## compute the norm of gradients
        total_norm = 0.
        for p in self.df.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        self.grad_norm = total_norm ** (1. / 2)

        if total_steps > 1000:
            nn.utils.clip_grad_norm_(self.df.parameters(), 0.3)
        
        self.optimizer.step()


    def get_current_errors(self):
        
        ret = OrderedDict([
            ('total', self.loss_total.data),
            ('simple', self.loss_simple.data),
            # ('vlb', self.loss_vlb.data),
            ('grad_norm', self.grad_norm)
        ])

        if hasattr(self, 'loss_gamma'):
            ret['gamma'] = self.loss_gamma.data

        return ret

    @torch.no_grad()
    def get_current_visuals(self):

        if self.isTrain:
            # self.img = self.img # input image
            self.img_gt = render_sdf(self.renderer, self.x) # rendered gt sdf
            self.img_gen_df = render_sdf(self.renderer, self.gen_df)

            self.img_bbox = render_bbox(self.renderer, self.bbox)
            meshes = [sdf_to_mesh_trimesh(self.gen_df[i][0]) for i in range(self.gen_df.shape[0])]
            vis_tensor_names = [
                # 'img',
                'img_gt',
                'img_gen_df',
                'img_bbox',
            ]
            vis_ims = self.tnsrs2ims(vis_tensor_names)
            visuals = zip(vis_tensor_names, vis_ims)
            visuals_dict = {
                "img": OrderedDict(visuals),
                "meshes": meshes,
                "paths": self.path,
                "bboxes": self.bbox,
            }
        else:
            self.img_gen_df = render_sdf(self.renderer, self.gen_df)
            self.img_bbox = render_bbox(self.renderer, self.bbox)
            meshes = [sdf_to_mesh_trimesh(self.gen_df[i][0]) for i in range(self.gen_df.shape[0])]
            vis_tensor_names = [
                'img_gen_df',
                'img_bbox',
            ]
            vis_ims = self.tnsrs2ims(vis_tensor_names)
            visuals = zip(vis_tensor_names, vis_ims)
            visuals_dict = {
                "img": OrderedDict(visuals),
                "meshes": meshes,
                "bboxes": self.bbox,
            }

        return visuals_dict

    def save(self, label, global_step, save_opt=False):
        state_dict = {
            'vqvae': self.vqvae_module.state_dict(),
            # 'cond_model': self.cond_model_module.state_dict(),
            'df': self.df_module.state_dict(),
            'global_step': global_step,
        }
        if save_opt:
            state_dict['opt'] = self.optimizer.state_dict()
        save_filename = 'df_%s.pth' % (label)
        save_path = os.path.join(self.opt.ckpt_dir, save_filename)

        torch.save(state_dict, save_path)

    def load_ckpt(self, ckpt, load_opt=False):

        # need this line or you will never be able to run inference code...
        map_fn = lambda storage, loc: storage
        if type(ckpt) == str:
            state_dict = torch.load(ckpt, map_location=map_fn)
        else:
            state_dict = ckpt

        self.vqvae.load_state_dict(state_dict['vqvae'])
        self.df.load_state_dict(state_dict['df'])
        # self.cond_model.load_state_dict(state_dict['cond_model'])
        print(colored('[*] weight successfully load from: %s' % ckpt, 'blue'))

        if load_opt:
            self.optimizer.load_state_dict(state_dict['opt'])
            print(colored('[*] optimizer successfully restored from: %s' % ckpt, 'blue'))


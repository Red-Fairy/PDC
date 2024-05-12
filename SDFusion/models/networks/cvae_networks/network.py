from __future__ import print_function

import torch
import torch.utils.data
from torch import nn
from torch.nn import init
from termcolor import colored, cprint
from torch.nn import functional as F

from einops import rearrange

from models.networks.ply_networks.pointnet2 import PointNet2
from models.networks.cvae_networks.cvae_modules import Encoder3D, Decoder3D

def init_weights(net, init_type='normal', gain=0.01):
    def init_func(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm2d') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                init.normal_(m.weight.data, 1.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=1.0)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == 'none':  # uses pytorch's default init method
                m.reset_parameters()
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    net.apply(init_func)

    # propagate to children
    for m in net.children():
        m.apply(init_func)


class CVAE(nn.Module):
    def __init__(self, ddconfig, condconfig, cond_ckpt):
        super(CVAE, self).__init__()

        self.device = 'cuda'
        self.ddconfig = ddconfig
        self,condconfig
        # self.embed_dim = self.ddconfig.cave.params._dim

        self.encoder = Encoder3D(**self.ddconfig)
        self.encoder_mu = nn.Sequential(
            
        )
        self.encoder_logvar = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 16 * 16 * 16, 3 * 16 * 16 * 16),
            nn.Unflatten(1, (3, 16, 16, 16))
        )
        self.encoder_logvar = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 16 * 16 * 16, 3 * 16 * 16 * 16),
            nn.Unflatten(1, (3, 16, 16, 16))
        )
        self.decoder = Decoder3D(**self.ddconfig)
        # self.bbox_cond_model = None
        self.ply_cond_model = PointNet2(hidden_dim=condconfig.context_dim).to(self.device)
        self.ply_cond_model.requires_grad_(True)
        load_result = self.ply_cond_model.load_state_dict(torch.load(cond_ckpt)['model_state_dict'], strict=False)
        print(load_result)
        print(colored('[*] conditional model successfully loaded', 'blue'))

        init_weights(self.encoder_mu, 'normal', 0.02)
        init_weights(self.encoder_logvar, 'normal', 0.02)
        init_weights(self.decoder, 'normal', 0.02)

    def _sampler(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """ Sampling with given mu and logvar
        
        Args:
            mu, logvar: Gaussian distribution parameters
        """
        var = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(var.size()).normal_()
        eps = eps.cuda()
        return eps.mul(var).add_(mu)

    def encode(self, x, ply, verbose=False):
        ply = self.ply_cond_model(ply)
        _z = self.encoder(x, cond_emb=ply)
        mu = self.encoder_mu(_z)
        logvar = self.encoder_logvar(_z)
        z = self._sampler(mu, logvar)
        if verbose:
            return z, mu, logvar
        else:
            return z
    
    def decode(self, z, ply):
        ply = self.ply_cond_model(ply)
        dec = self.decoder(z, cond_emb=ply)
        return dec
    
    def forward(self, input, verbose=False, encode_only=False):
        raise NotImplementedError
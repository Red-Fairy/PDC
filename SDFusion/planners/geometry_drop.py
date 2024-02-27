import os
import copy
from collections import OrderedDict

import numpy as np
from omegaconf import OmegaConf
from termcolor import colored, cprint

import numpy as np
import mcubes
from omegaconf import OmegaConf
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable

from planners.base_model import BasePlanner

from utils.util_3d import sdf_to_mesh, sdf_to_mesh_trimesh, mesh_to_sdf


class GeometryDropPlanner(BasePlanner):
    def name(self):
        return 'GeometryDropPlanner'
    
    def initialize(self, opt):
        BasePlanner.initialize(self, opt)
        self.opt = opt
        self.model_name = self.name()
        self.device = opt.device

    ## softloss planning
    # def plan(self, ori_sdf, t):
        # B, _, res, _, _ = ori_sdf.shape
        # ori_sdf = ori_sdf.view(B, res, res, res).detach().cpu().numpy()

        # ## clip sdf
        # pro_sdf = []
        # with torch.enable_grad():
        #     for i in range(B):
        #         i_sdf = copy.deepcopy(ori_sdf[i])
        #         ## 10.16 constraint, detail see GoogleSlides
        #         # i_sdf[:50, :, :16] = 0.1
        #         # i_sdf[:50, :, -16:] = 0.1
                
        #         ## define optimizer
        #         i_sdf = torch.tensor(i_sdf, requires_grad=True, device=self.device)
        #         # i_sdf = torch.randn(64, 64, 64, requires_grad=True, device='cuda')
        #         loss_pen = (F.relu( - i_sdf)).sum()
        #         loss_pen.backward()
        #         optimizer = optim.Adam([i_sdf], lr=0.01)

        #         ## optimize i_sdf
        #         for j in range(5):
        #             # print(j)
        #             loss_pen = torch.sum(F.relu(0.05 - i_sdf[:, :, :16])) + \
        #                 torch.sum(F.relu(0.05 - i_sdf[:, :, -16:]))
        #             # loss_pen = torch.sum(F.relu(0.05 - i_sdf[:, :, :16])) + torch.sum(F.relu(0.05 - i_sdf[:, :, -16:]))
        #             loss_pen.backward()
        #             # print(f'after BN: {j}')
        #             optimizer.step()
        #             optimizer.zero_grad()
        #             cprint(f'[*] iter: {j} . loss: {loss_pen.item()}', 'blue')

        #         i_sdf = i_sdf.detach().cpu().numpy()
        #         pro_sdf.append(i_sdf)

        # pro_sdf = np.stack(pro_sdf, axis=0)
        # pro_sdf = torch.Tensor(pro_sdf).to(self.device).unsqueeze(1)
        # return pro_sdf

    ## hardreplace planning
    def plan(self, ori_sdf, t):
        B, _, res, _, _ = ori_sdf.shape
        ori_sdf = ori_sdf.view(B, res, res, res).detach().cpu().numpy()

        ## clip sdf
        pro_sdf = []
        # with torch.enable_grad():
        for i in range(B):
            i_sdf = copy.deepcopy(ori_sdf[i])
            ## 10.20 hardreplacement
            i_sdf[:50, :, :16] = 0.1
            i_sdf[:50, :, -16:] = 0.1
            
            i_mesh = sdf_to_mesh_trimesh(i_sdf, level=0.02)
            i_sdf = mesh_to_sdf(i_mesh)
            i_sdf = np.clip(i_sdf, -0.2, 0.2)

            pro_sdf.append(i_sdf)

        pro_sdf = np.stack(pro_sdf, axis=0)
        pro_sdf = torch.Tensor(pro_sdf).to(self.device).unsqueeze(1)
        return pro_sdf

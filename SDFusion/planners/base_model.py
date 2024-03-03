import os
from termcolor import colored, cprint
import torch
import utils.util as util

def create_planner(opt):
    if opt.planner == 'GeometryDropPlanner':
        from planners.geometry_drop import GeometryDropPlanner
        planner = GeometryDropPlanner(opt)

    else:
        planner = BasePlanner(opt)
    
    # elif opt.planner == 'BasePlanner':
    #     planner = BasePlanner()

    # else:
    #     raise ValueError("Planner [%s] not recognized." % opt.planner)
    
    cprint("[*] Planner has been created: %s" % planner.name(), 'blue')
    return planner


class BasePlanner():
    def name(self):
        return 'BasePlanner'
    
    def __init__(self, opt, device='cuda'):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.device = device
        # self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor

    def plan(self, ori_sdf, t):
        return ori_sdf
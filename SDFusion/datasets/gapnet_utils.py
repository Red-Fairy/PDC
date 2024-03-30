import os.path
import json

import h5py
import numpy as np
from termcolor import colored, cprint

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode

from datasets.base_dataset import BaseDataset
from omegaconf import OmegaConf

import open3d
import trimesh
from datasets.convert_utils import mesh_to_sdf

def pc_normalize(pc, scale_norm=True, return_stat=False):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    ret = {'centroid': torch.from_numpy(centroid).float(), 'rotation': torch.eye(4).float()}
    if scale_norm:
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        ret['scale'] = torch.from_numpy(m)
    if return_stat:
        return pc, ret
    else:
        return pc

def get_single_model(opt):

    basepath = opt.dataroot
    cat = opt.cat
    model_id = opt.model_id

    ret = {}

    sdf_path = os.path.join(basepath, cat , model_id + '.h5')
    ret['path'] = [sdf_path]
    h5_f = h5py.File(sdf_path, 'r')
    sdf = h5_f['pc_sdf_sample'][:].astype(np.float32)
    ret['sdf'] = torch.Tensor(sdf).view(1, 1, opt.res, opt.res, opt.res)

    cond_path = sdf_path.replace('part_sdf', 'part_ply_fps').replace('.h5', '.ply')
    points = np.array(open3d.io.read_point_cloud(cond_path).points)
    points, points_stat = pc_normalize(points, scale_norm=False, return_stat=True)
    points = torch.from_numpy(points).transpose(0, 1).float().unsqueeze(0) # (1, 3, N)
    ret['ply'] = points
    ret['ply_translation'] = points_stat['centroid'].unsqueeze(0) # (1, 3)

    transform_path = sdf_path.replace('part_sdf', 'part_bbox_aligned').replace('.h5', '.json')
    with open(transform_path, 'r') as f:
        transform = json.load(f)
        part_translate, part_extent = torch.tensor(transform['centroid']).float(), torch.tensor(transform['extents']).float()
    ret['part_translation'] = part_translate.unsqueeze(0) # (1, 3)
    ret['part_extent'] = part_extent.unsqueeze(0) # (1, 3)

    if opt.use_mobility_constraint:
        mobility_path = sdf_path.replace('part_sdf', 'part_mobility').replace('.h5', '.json')
        with open(mobility_path, 'r') as f:
            mobility = json.load(f)
            move_axis, move_limit = torch.tensor(mobility['move_axis']).float(), torch.tensor(mobility['move_limit']).float()
        ret['move_axis'] = move_axis
        ret['move_limit'] = move_limit

    if opt.initial_shape_path is not None:
        if opt.initial_shape_path != 'cheat':
            initial_mesh = trimesh.load_mesh(opt.initial_shape_path)
            sdf = mesh_to_sdf(initial_mesh, res=opt.res, padding=0.2, trunc=opt.trunc_thres, device='cuda')
            ret['initial_shape'] = sdf.unsqueeze(0)
        else: # cheating for debug propose, use the ground truth shape
            ret['initial_shape'] = ret['sdf']

    return ret

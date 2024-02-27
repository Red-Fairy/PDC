"""
    adopted from: https://github.com/shubhtuls/PixelTransformer/blob/03b65b8612fe583b3e35fc82b446b5503dd7b6bd/data/shapenet.py
"""
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

# from https://github.com/laughtervv/DISN/blob/master/preprocessing/info.json
class GAPartNetDataset(BaseDataset):

    def __init__(self, opt, phase='train', cat='all', res=256, eval_samples=100):
        self.phase = phase
        self.opt = opt
        self.max_dataset_size = opt.max_dataset_size
        self.res = res
        dataroot = opt.dataroot

        self.sdf_filepaths = [os.path.join(dataroot, f) for f in os.listdir(dataroot) if f.endswith('.h5')]
        self.sdf_filepaths = list(filter(lambda f: os.path.exists(f.replace('part_sdf', 'part_ply').replace('.h5', '.ply')), self.sdf_filepaths))

        self.bbox_cond = opt.bbox_cond
        self.ply_cond = opt.ply_cond
        self.df_conf = OmegaConf.load(opt.df_cfg)

        if self.phase == 'eval':
            # generate some random bbox for evaluation
            vertices = torch.Tensor([[1,1,-1],[1,1,1],[-1,1,1],[-1,1,-1],[1,-1,-1],[1,-1,1],[-1,-1,1],[-1,-1,-1]])
            self.bbox_list = []
            for _ in range(eval_samples):
                bbox = torch.rand(3) * 0.3 + 0.2
                bbox_full = vertices * bbox.view(1, 3).expand(8, 3)
                self.bbox_list.append(bbox_full)
            self.N = eval_samples
            return

        self.sdf_filepaths = self.sdf_filepaths[:self.max_dataset_size]
        cprint('[*] %d samples loaded.' % (len(self.sdf_filepaths)), 'yellow')

        self.N = len(self.sdf_filepaths)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


    def __getitem__(self, index):

        if self.phase == 'eval':
            return {'bbox': self.bbox_list[index]}

        sdf_h5_file = self.sdf_filepaths[index]

        h5_f = h5py.File(sdf_h5_file, 'r')
        sdf = h5_f['pc_sdf_sample'][:].astype(np.float32)
        sdf = torch.Tensor(sdf).view(1, self.res, self.res, self.res)

        thres = self.opt.trunc_thres
        if thres != 0.0:
            sdf = torch.clamp(sdf, min=-thres, max=thres)

        ret = {
            'sdf': sdf,
            'path': sdf_h5_file,
        }

        if self.bbox_cond:
            bbox_filepath = sdf_h5_file.replace('part_sdf', 'part_bbox').replace('.h5', '.npy')
            bbox = torch.tensor(np.load(bbox_filepath))
            ret['bbox'] = bbox

        if self.ply_cond:
            ply_filepath = sdf_h5_file.replace('part_sdf', 'part_ply').replace('.h5', '.ply')
            ret['ply'] = ply_filepath
            # load ply file
            points = torch.from_numpy(np.array(open3d.io.read_point_cloud(ply_filepath).points)).transpose(0, 1).float() # (3, N)
            # padding
            N = points.shape[1]
            points = torch.cat([points, torch.zeros(3, self.df_conf.ply.max_points - N)], dim=1)
            ret['ply'] = points

        return ret

    def __len__(self):
        return self.N

    def name(self):
        return 'GAPartNetSDFDataset'
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
import trimesh
from datasets.mesh_to_sdf import mesh_to_sdf
from datasets.gapnet_utils import pc_normalize

class GAPartNetDataset(BaseDataset):

    def __init__(self, opt, phase='train', cat='all', res=256, eval_samples=100):
        self.phase = phase
        self.opt = opt
        self.max_dataset_size = opt.max_dataset_size
        self.res = res
        dataroot = opt.dataroot

        if self.phase == 'refine':
            assert opt.batch_size == 1

        self.sdf_filepaths = [os.path.join(dataroot, f) for f in os.listdir(dataroot) if f.endswith('.h5')]
        self.sdf_filepaths = list(filter(lambda f: os.path.exists(f.replace('part_sdf', 'part_ply').replace('.h5', '.ply')), self.sdf_filepaths))

        self.bbox_cond = opt.bbox_cond
        self.joint_rotate = opt.joint_rotate

        self.ply_cond = opt.ply_cond
        self.ply_input_rotate = opt.ply_input_rotate
        assert not (self.ply_input_rotate and self.joint_rotate), "ply_rot and joint_rotate cannot be both True"

        self.df_conf = OmegaConf.load(opt.df_cfg)

        # if self.phase == 'eval':
        #     # generate some random bbox for evaluation
        #     vertices = torch.Tensor([[1,1,-1],[1,1,1],[-1,1,1],[-1,1,-1],[1,-1,-1],[1,-1,1],[-1,-1,1],[-1,-1,-1]])
        #     self.bbox_list = []
        #     for _ in range(eval_samples):
        #         bbox = torch.rand(3) * 0.3 + 0.2
        #         bbox_full = vertices * bbox.view(1, 3).expand(8, 3)
        #         self.bbox_list.append(bbox_full)
        #     self.N = eval_samples
        #     return

        self.sdf_filepaths = self.sdf_filepaths[:self.max_dataset_size]
        cprint('[*] %d samples loaded.' % (len(self.sdf_filepaths)), 'yellow')

        self.N = len(self.sdf_filepaths)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


    def __getitem__(self, index):

        sdf_h5_file = self.sdf_filepaths[index]

        ret = {'path': sdf_h5_file}

        if not self.joint_rotate: # if joint rotate, 'sdf' will be calculated on the fly

            h5_f = h5py.File(sdf_h5_file, 'r')
            sdf = h5_f['pc_sdf_sample'][:].astype(np.float32)
            sdf = torch.Tensor(sdf).view(1, self.res, self.res, self.res)

            thres = self.opt.trunc_thres
            if thres != 0.0:
                sdf = torch.clamp(sdf, min=-thres, max=thres)
            ret['sdf'] = sdf

        if self.bbox_cond:
            bbox_filepath = sdf_h5_file.replace('part_sdf', 'part_bbox').replace('.h5', '.npy')
            bbox = torch.tensor(np.load(bbox_filepath))
            if self.joint_rotate:
                bbox = torch.mm(bbox, torch.tensor(rot_matrix).float())
            ret['bbox'] = bbox

        if self.ply_cond:
            ply_filepath = sdf_h5_file.replace('part_sdf', 'part_ply_fps').replace('.h5', '.ply')
            # load ply file
            ply_file = open3d.io.read_point_cloud(ply_filepath).points
            points = np.array(ply_file)
            points, points_stat = pc_normalize(points, scale_norm=False, return_stat=True)
            points = torch.from_numpy(points).transpose(0, 1).float() # (3, N)

            transform_path = sdf_h5_file.replace('part_sdf', 'part_bbox_aligned').replace('.h5', '.json')
            with open(transform_path, 'r') as f:
                transform = json.load(f)
                part_translate, part_extent = torch.from_numpy(np.array(transform['centroid'])).float(), torch.from_numpy(np.array(transform['extents'])).float()

            if self.ply_input_rotate:
                raw, pitch, yaw = torch.rand(3) * 2 * np.pi
                rot_matrix = torch.tensor([
                    [np.cos(yaw)*np.cos(pitch), np.cos(yaw)*np.sin(pitch)*np.sin(raw)-np.sin(yaw)*np.cos(raw), np.cos(yaw)*np.sin(pitch)*np.cos(raw)+np.sin(yaw)*np.sin(raw)],
                    [np.sin(yaw)*np.cos(pitch), np.sin(yaw)*np.sin(pitch)*np.sin(raw)+np.cos(yaw)*np.cos(raw), np.sin(yaw)*np.sin(pitch)*np.cos(raw)-np.cos(yaw)*np.sin(raw)],
                    [-np.sin(pitch), np.cos(pitch)*np.sin(raw), np.cos(pitch)*np.cos(raw)]
                ])
                points = torch.mm(rot_matrix, points) # (3, N)
                points_stat['centroid'] = torch.mm(rot_matrix, points_stat['centroid'].view(3, 1)).view(3)

            if self.joint_rotate:
                # rotate the sdf, must re-build from the mesh
                mesh_path = sdf_h5_file.replace('part_sdf', 'part_meshes').replace('.h5', '.obj')
                mesh = trimesh.load_mesh(mesh_path)

                raw, pitch, yaw = torch.rand(3) * 2 * np.pi
                rot_matrix = torch.tensor([
                    [np.cos(yaw)*np.cos(pitch), np.cos(yaw)*np.sin(pitch)*np.sin(raw)-np.sin(yaw)*np.cos(raw), np.cos(yaw)*np.sin(pitch)*np.cos(raw)+np.sin(yaw)*np.sin(raw), 0],
                    [np.sin(yaw)*np.cos(pitch), np.sin(yaw)*np.sin(pitch)*np.sin(raw)+np.cos(yaw)*np.cos(raw), np.sin(yaw)*np.sin(pitch)*np.cos(raw)-np.cos(yaw)*np.sin(raw), 0],
                    [-np.sin(pitch), np.cos(pitch)*np.sin(raw), np.cos(pitch)*np.cos(raw), 0],
                    [0, 0, 0, 1]
                ])

                points = torch.cat([points, torch.ones(1, points.shape[1])], dim=0)
                points = torch.mm(rot_matrix, points)[:-1]
                points_stat['centroid'] = torch.mm(rot_matrix[:3, :3], points_stat['centroid'].view(3, 1)).view(3)

                # rotate the mesh
                mesh.apply_transform(rot_matrix)
                part_translate = torch.from_numpy(np.array(mesh.bounding_box.centroid)).float()
                part_extent = torch.from_numpy(np.array(mesh.bounding_box.extents)).float()

                # move the mesh to the origin
                mesh.apply_translation(-mesh.bounding_box.centroid)
                mesh.apply_scale(1. / np.max(np.abs(mesh.bounds)))

                # convert to sdf
                sdf = mesh_to_sdf(mesh, self.res, trunc=self.opt.trunc_thres, padding=0.2)
                ret['sdf'] = sdf
        
            # padding
            N = points.shape[1]
            # points = torch.cat([points, torch.zeros(3, self.df_conf.ply.max_points - N)], dim=1)
            ret['ply'] = points
            ret['ply_translation'] = points_stat['centroid']
            ret['part_translation'] = part_translate
            ret['part_extent'] = part_extent

        return ret

    def __len__(self):
        return self.N

    def name(self):
        return 'GAPartNetSDFDataset'
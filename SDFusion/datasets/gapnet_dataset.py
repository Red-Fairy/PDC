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
from utils.util_3d import mesh_to_sdf
from datasets.gapnet_utils import pc_normalize
import random

import glob

def build_rot_matrix(rotate_angle_y):
    rot_matrix = torch.tensor([
        [np.cos(rotate_angle_y), 0, np.sin(rotate_angle_y), 0],
        [0, 1, 0, 0],
        [-np.sin(rotate_angle_y), 0, np.cos(rotate_angle_y), 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32)
    return rot_matrix

class GAPartNetDataset(BaseDataset):

    def __init__(self, opt, phase='train', cat='all', res=256, eval_samples=100, haoran=False, haoran_rotation=False):
        self.phase = phase
        self.opt = opt
        self.max_dataset_size = opt.max_dataset_size
        self.res = res
        self.haoran = haoran
        self.haoran_rotation = haoran_rotation
        if self.haoran:
            self.haoran_override = f'../data_lists/pred_pose_regression/{cat}/set_{opt.testset_idx}'

        if self.phase == 'refine':
            assert opt.batch_size == 1

        self.sdf_dir = f'{opt.sdf_mode}_sdf_{opt.res}'
        self.pcd_dir = opt.pcd_dir # part_ply_fps

        if cat != 'all':
            dataroot = os.path.join(opt.dataroot, self.sdf_dir, cat)
            self.sdf_filepaths = [os.path.join(dataroot, f) for f in os.listdir(dataroot) if f.endswith('.h5')]
            # only test the following ids
            # ids = ['22301', '23372', '25144']
            # self.sdf_filepaths = [f for f in self.sdf_filepaths if any([i in f for i in ids])]
            if opt.sdf_mode == 'part' and opt.model_id is None:
                # (self.phase == 'train' or self.phase == 'test'):
                filelist_path = opt.dataroot.replace('dataset', 'data_lists/'+phase)
                with open(os.path.join(filelist_path, cat+'.txt'), 'r') as f:
                    file_names = [line.strip() for line in f]
                self.sdf_filepaths = [f for f in self.sdf_filepaths if f.split('/')[-1].split('.')[0] in file_names]
            elif opt.model_id is not None:
                self.sdf_filepaths = [f for f in self.sdf_filepaths if any([i in f for i in opt.model_id])]
                print(self.sdf_filepaths)
            self.sdf_filepaths = list(filter(lambda f: os.path.exists(f.replace(self.sdf_dir, self.pcd_dir).replace('.h5', '.ply')), self.sdf_filepaths))
        else:
            self.sdf_filepaths = []
            for c in os.listdir(os.path.join(opt.dataroot, self.sdf_dir)):
                cat_dir = os.path.join(opt.dataroot, self.sdf_dir, c)
                cat_files = [os.path.join(cat_dir, f) for f in os.listdir(cat_dir) if f.endswith('.h5')]
                self.sdf_filepaths.extend(cat_files)
                extend_scale = max(int(5000 / len(cat_files)), 1)
                self.sdf_filepaths.extend(cat_files * extend_scale)
                print('Extend scale for %s: %d, ori len %d' % (c, extend_scale, len(cat_files)))

        if self.haoran or self.haoran_rotation:
            self.sdf_filepaths = list(filter(lambda f: os.path.exists(os.path.join(self.haoran_override, os.path.basename(f).replace('.h5', '.json'))), self.sdf_filepaths))
            
        # self.bbox_cond = opt.bbox_cond

        self.ply_cond = opt.ply_cond
        self.joint_rotate = opt.joint_rotate
        self.ply_rotate = opt.ply_rotate

        self.ply_bbox_cond = opt.ply_bbox_cond

        # self.df_conf = OmegaConf.load(opt.df_cfg)

        self.sdf_filepaths = self.sdf_filepaths[:self.max_dataset_size]
        self.sdf_filepaths = sorted(self.sdf_filepaths)
        if self.opt.start_idx is not None and self.opt.end_idx is not None:
            self.sdf_filepaths = self.sdf_filepaths[self.opt.start_idx: min(self.opt.end_idx, len(self.sdf_filepaths))]
        cprint('[*] %d samples loaded.' % (len(self.sdf_filepaths)), 'yellow')

        if not self.opt.isTrain and self.opt.test_diversity: # repeat the dataset for diversity testing
            self.sdf_filepaths = [x for x in self.sdf_filepaths for _ in range(self.opt.diversity_count)]

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __getitem__(self, index):

        sdf_h5_file = self.sdf_filepaths[index]
        # print(sdf_h5_file)

        ret = {'path': sdf_h5_file}

        if not self.joint_rotate: # if joint rotate, 'sdf' will be calculated on the fly
            h5_f = h5py.File(sdf_h5_file, 'r')
            sdf = h5_f['pc_sdf_sample'][:].astype(np.float32)
            sdf = torch.Tensor(sdf).view(1, self.res, self.res, self.res)

            thres = self.opt.trunc_thres
            if thres != 0.0:
                sdf = torch.clamp(sdf, min=-thres, max=thres)
            ret['sdf'] = sdf

        if not self.opt.isTrain and self.phase == 'test' and self.opt.use_bbox_mesh:
            bbox_filepath = sdf_h5_file.replace(self.sdf_dir, 'bbox_mesh').replace('.h5', '.obj')
            mesh_sdf = mesh_to_sdf(trimesh.load_mesh(bbox_filepath), self.res, trunc=self.opt.trunc_thres, padding=0.2)
            ret['bbox_mesh'] = mesh_sdf

        # if self.bbox_cond:
        #     bbox_filepath = sdf_h5_file.replace(self.sdf_dir, 'part_bbox').replace('.h5', '.npy')
        #     bbox = torch.tensor(np.load(bbox_filepath))
        #     ret['bbox'] = bbox

        if self.ply_cond or self.ply_bbox_cond:
            ply_filepath = sdf_h5_file.replace(self.sdf_dir, self.pcd_dir).replace('.h5', '.ply')
            ply_file = open3d.io.read_point_cloud(ply_filepath).points

            points = np.array(ply_file)
            points, points_stat = pc_normalize(points, scale_norm=self.opt.ply_norm, return_stat=True)
            points = torch.from_numpy(points).transpose(0, 1).float() # (3, N)

            transform_path = sdf_h5_file.replace(self.sdf_dir, 'part_translation_scale').replace('.h5', '.json')
            with open(transform_path, 'r') as f:
                transform = json.load(f)
                part_translate, part_extent = torch.tensor(transform['centroid']).float(), torch.tensor(transform['extents']).float()
                if self.opt.use_predicted_scale:
                    predicted_scale_path = sdf_h5_file.replace(self.sdf_dir, 'part_scale_predicted').replace('.h5', '.json')
                    part_extent = torch.tensor(json.load(open(predicted_scale_path))['scale']).float().view(1, 1).repeat(1, 3)
                if self.opt.use_predicted_volume:
                    predicted_volume_path = sdf_h5_file.replace(self.sdf_dir, 'part_volume_predicted').replace('.h5', '.json') # cube root of the volume
                    part_extent = torch.tensor(json.load(open(predicted_volume_path))['volume']).float().view(1, 1).repeat(1, 3)

            if self.haoran or self.haoran_rotation:
                transform_path = os.path.join(self.haoran_override, os.path.basename(ply_filepath).replace('.ply', '.json'))
                with open(transform_path, 'r') as f:
                    transform = json.load(f)
                    part_translate_pred, part_extent_pred = torch.tensor(transform['centroid']).float(), torch.tensor(transform['extents']).float()
                    ret['part_translation_pred'] = part_translate_pred
                    ret['part_extent_pred'] = part_extent_pred
                
            if self.opt.use_mobility_constraint and self.opt.cat in ['slider_drawer', 'hinge_door']:
                if self.haoran:
                    if self.opt.cat == 'slider_drawer':
                        move_axis = torch.tensor([0, 0, 1], dtype=torch.float32)
                        move_limit = torch.tensor([0, 0.2], dtype=torch.float32)
                        move_origin = part_translate_pred + part_extent_pred * 0.5
                    elif self.opt.cat == 'hinge_door':
                        move_axis = torch.tensor([1, 0, 0], dtype=torch.float32)
                        move_limit = torch.tensor([0, 90], dtype=torch.float32)
                        move_origin = part_translate_pred + part_extent_pred * 0.5
                else:
                    mobility_path = sdf_h5_file.replace(self.sdf_dir, 'part_mobility').replace('.h5', '.json')
                    with open(mobility_path, 'r') as f:
                        mobility = json.load(f)
                        move_axis, move_limit, move_origin = \
                            torch.tensor(mobility['move_axis']).float(), torch.tensor(mobility['move_limit']).float(), torch.tensor(mobility['move_origin']).float()
                ret['move_axis'] = move_axis
                ret['move_limit'] = move_limit
                ret['move_origin'] = move_origin

            if self.ply_rotate: # only rotate the point cloud condition
                
                if self.haoran or self.haoran_rotation:
                    rotate_angle_y = transform['rotate_angle'] * np.pi / 180
                elif self.opt.rotate_angle is None:
                    rotate_angle_y = random.random() * 2 * np.pi
                else:
                    rotate_angle_y = self.opt.rotate_angle * np.pi / 180

                rot_matrix = build_rot_matrix(rotate_angle_y)

                points = torch.mm(rot_matrix, torch.cat([points, torch.ones(1, points.shape[1])], dim=0))[:-1]

                points_stat['rotation'] = rot_matrix

                if self.haoran: # use the predicted rotation matrix
                    ret['ply_rotation_pred'] = build_rot_matrix(transform['rotate_angle_pred'] * np.pi / 180)
            
            else:
                points_stat['rotation'] = torch.eye(4)

                # points_stat['centroid'] = torch.mm(rot_matrix, points_stat['centroid'].view(3, 1)).view(3)

            if self.joint_rotate: # jointly rotate the mesh and point cloud condition, calculate mesh on the fly

                assert False # not used now

                mesh_path = sdf_h5_file.replace(self.sdf_dir, 'part_meshes').replace('.h5', '.obj')
                mesh = trimesh.load_mesh(mesh_path)

                rotate_angle_y = torch.rand(1) * 2 * np.pi
                rot_matrix = torch.tensor([
                    [np.cos(rotate_angle_y), 0, np.sin(rotate_angle_y), 0],
                    [0, 1, 0, 0],
                    [-np.sin(rotate_angle_y), 0, np.cos(rotate_angle_y), 0],
                    [0, 0, 0, 1]
                ])

                mesh.apply_transform(rot_matrix)
                # convert to sdf
                sdf = mesh_to_sdf(mesh, self.res, trunc=self.opt.trunc_thres, padding=0.2) # scale the mesh to unit cube with padding 0.2 (max_extent=4/2.2)
                ret['sdf'] = sdf

                points = torch.mm(rot_matrix, torch.cat([points, torch.ones(1, points.shape[1])], dim=0))[:-1]
                points_stat['rotation'] = rot_matrix
                # points_stat['centroid'] = torch.mm(rot_matrix[:3, :3], points_stat['centroid'].view(3, 1)).view(3)

                part_translate = torch.tensor(mesh.bounding_box.centroid).float()
                part_extent = torch.tensor(mesh.bounding_box.extents).float()
        
            # padding
            # N = points.shape[1]
            # points = torch.cat([points, torch.zeros(3, self.df_conf.ply.max_points - N)], dim=1)
            ret['ply'] = points
            ret['ply_translation'] = points_stat['centroid']
            ret['ply_rotation'] = points_stat['rotation']
            ret['ply_scale'] = points_stat['scale'].view(1)

            ret['part_translation'] = part_translate 
            ret['part_extent'] = part_extent

        return ret

    def __len__(self):
        return len(self.sdf_filepaths)

    def name(self):
        return 'GAPartNetSDFDataset'


class GAPartNetDataset4ScalePrediction(BaseDataset):

    def __init__(self, opt, phase='train', cat='all', extend_size=None):

        self.phase = phase
        self.pcd_dir = f'part_ply_fps'

        dataroot = os.path.join(opt.dataroot, self.pcd_dir, cat)
        self.filepaths = [os.path.join(dataroot, x) for x in os.listdir(dataroot) if x.endswith('.ply')]

        if self.phase == 'train' or self.phase == 'test':
            filelist_path = opt.dataroot.replace('dataset', 'data_lists/'+phase)
            with open(os.path.join(filelist_path, cat+'.txt'), 'r') as f:
                file_names = [line.strip() for line in f]
            self.filepaths = [f for f in self.filepaths if f.split('/')[-1].split('.')[0] in file_names]

        cprint('[*] %d samples loaded.' % (len(self.filepaths)), 'yellow')

        # extend the dataset size
        if extend_size is not None:
            self.filepaths = (self.filepaths * (extend_size // len(self.filepaths) + 1))[:extend_size]

        self.ply_rotate = opt.ply_rotate

    def __getitem__(self, index):

        ply_filepath = self.filepaths[index]

        ret = {'path': ply_filepath}

        # load ply file
        ply_file = open3d.io.read_point_cloud(ply_filepath).points
        points = np.array(ply_file)
        points, points_stat = pc_normalize(points, scale_norm=True, return_stat=True)
        points = torch.from_numpy(points).transpose(0, 1).float() # (3, N)

        transform_path = ply_filepath.replace('part_ply_fps', 'part_translation_scale').replace('.ply', '.json')
        with open(transform_path, 'r') as f:
            transform = json.load(f)
            _, part_extent = torch.tensor(transform['centroid']).float(), torch.tensor(transform['extents']).float()

        if self.ply_rotate:

            rotate_angle_y = torch.rand(1) * 2 * np.pi 
            rot_matrix = torch.tensor([
                [np.cos(rotate_angle_y), 0, np.sin(rotate_angle_y), 0],
                [0, 1, 0, 0],
                [-np.sin(rotate_angle_y), 0, np.cos(rotate_angle_y), 0],
                [0, 0, 0, 1]
            ])

            points = torch.mm(rot_matrix, torch.cat([points, torch.ones(1, points.shape[1])], dim=0))[:-1]
            # points_stat['rotation'] = rot_matrix

        ret['ply'] = points
        ret['ply_scale'] = points_stat['scale'].view(1)
        ret['part_extent'] = part_extent

        return ret

    def __len__(self):
        return len(self.filepaths)

    def name(self):
        return 'GAPartNet4ScalePrediction'
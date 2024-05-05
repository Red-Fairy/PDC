import open3d
import os
from pytorch3d.loss import chamfer_distance
import torch
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--gt_root', type=str, default='/mnt/azureml/cr/j/19c62471467141d39f5f0dc988c1ea42/exe/wd/data-rundong/PartDiffusion/dataset/part_meshes/slider_drawer')
parser.add_argument('--test_root', type=str, default='/mnt/azureml/cr/j/19c62471467141d39f5f0dc988c1ea42/exe/wd/PartDiffusion/SDFusion/logs/slider-ply2shape-plyrot-scale3-lr0.00001/test_250000_rotate0.0_scale3.0_eta0.0_steps50_volume_mobility_diversity_margin128-haoran/selected')
args = parser.parse_args()

# run in SDFusion directory
gt_root = args.gt_root
test_root = args.test_root

obj_files_test = sorted([os.path.join(test_root, f) for f in os.listdir(test_root) if f.endswith('.obj')])

obj_files_gt = sorted([os.path.join(gt_root, f) for f in os.listdir(test_root) if f.endswith('.obj')])

num_points = 10000

for (obj_file_test, obj_file_gt) in zip(obj_files_test, obj_files_gt):
    assert obj_file_test.split('/')[-1] == obj_file_gt.split('/')[-1]
    obj_test = open3d.io.read_triangle_mesh(obj_file_test)
    obj_gt = open3d.io.read_triangle_mesh(obj_file_gt)
    pcd_test = obj_test.sample_points_uniformly(number_of_points=num_points) # 
    pcd_gt = obj_gt.sample_points_uniformly(number_of_points=num_points)
    pointclouds_test = torch.tensor(np.asarray(pcd_test.points)).to('cuda').unsqueeze(0) # (1, num_points, 3)
    pointclouds_gt = torch.tensor(np.asarray(pcd_gt.points)).to('cuda').unsqueeze(0) # (1, num_points, 3)
    loss = chamfer_distance(pointclouds_test, pointclouds_gt, batch_reduction='sum', point_reduction='mean')[0]
    print(loss)


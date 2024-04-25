import open3d
import os
from pytorch3d.loss import chamfer_distance
import torch
import numpy as np

root = '/raid/haoran/Project/PartDiffusion/PartDiffusion/SDFusion/logs/slider-ply2shape-plyrot-scale3-lr0.00001/test_diversity_200000_rotate0.0_scale3.0_eta0.0_steps50'

obj_files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.obj')]

num_points = 10000

pointclouds = torch.zeros((len(obj_files), num_points, 3))

for i, obj_file in enumerate(obj_files):
    obj = open3d.io.read_triangle_mesh(obj_file)
    pcd = obj.sample_points_uniformly(number_of_points=num_points)
    pointclouds[i] = torch.tensor(np.asarray(pcd.points))

# calculate the chamfer distance between each pair of point clouds
loss = 0.

pointclouds = pointclouds.to('cuda')

for i in range(1, len(obj_files)):
    rearrange = [len(obj_files) - i + j for j in range(i)] + [i for i in range(len(obj_files) - i)]
    pointclouds_i = pointclouds[rearrange]
    loss += chamfer_distance(pointclouds, pointclouds_i, batch_reduction='sum', point_reduction='mean')[0]

loss /= len(obj_files) * (len(obj_files) - 1)

print(loss)

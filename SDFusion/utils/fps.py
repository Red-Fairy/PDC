from fpsample import bucket_fps_kdtree_sampling
import numpy as np
import open3d
import os
from tqdm import tqdm

cat = 'hinge_door'

root = f'/raid/haoran/Project/PartDiffusion/PartDiffusion/dataset/part_ply/{cat}'
save_root = f'/raid/haoran/Project/PartDiffusion/PartDiffusion/dataset/part_ply_fps/{cat}'

os.makedirs(save_root, exist_ok=True)

ply_files = sorted(os.listdir(root))

ply_files = [os.path.join(root, ply_file) for ply_file in ply_files if ply_file.endswith('.ply')]

# N_point_min = 10000

# point_dict = {}

# for ply_file in ply_files:
#     id = ply_file.split('/')[-1].split('.')[0].split('_')[0]
#     part_id = ply_file.split('/')[-1].split('.')[0].split('_')[1]
#     pcd = open3d.io.read_point_cloud(ply_file)
#     points = np.array(pcd.points)
#     point_dict[f'{id}_{part_id}'] = points.shape[0]
#     N_point_min = min(N_point_min, points.shape[0])

# print(N_point_min)
# sort by number of points

    
N_point = 5500 

for ply_file in tqdm(ply_files):
    
    id = ply_file.split('/')[-1].split('.')[0].split('_')[0]
    part_id = ply_file.split('/')[-1].split('.')[0].split('_')[1]

    # read point cloud
    pcd = open3d.io.read_point_cloud(ply_file)

    # downsample
    fps_samples_idx = bucket_fps_kdtree_sampling(np.array(pcd.points), N_point)
    downsampled_pcd = open3d.geometry.PointCloud()
    downsampled_pcd.points = open3d.utility.Vector3dVector(np.array(pcd.points)[fps_samples_idx])

    # save
    save_path = os.path.join(save_root, f'{id}_{part_id}.ply')
    open3d.io.write_point_cloud(save_path, downsampled_pcd)

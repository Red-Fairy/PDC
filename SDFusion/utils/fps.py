from fpsample import bucket_fps_kdtree_sampling
import numpy as np
import open3d
import os
from tqdm import tqdm

def main():
    categories = ['line_fixed_handle', 'hinge_handle', 'hinge_knob', 'hinge_lid', 'round_fixed_handle']

    for cat in categories:

        print(cat)
        
        root = f'/raid/haoran/Project/PartDiffusion/PartDiffusion/dataset/part_ply/{cat}'
        save_root = f'/raid/haoran/Project/PartDiffusion/PartDiffusion/dataset/part_ply_fps/{cat}'

        os.makedirs(save_root, exist_ok=True)

        ply_files = sorted(os.listdir(root))

        ply_files = [os.path.join(root, ply_file) for ply_file in ply_files if ply_file.endswith('.ply')]
            
        N_point = 10000

        for ply_file in tqdm(ply_files):
            
            object_id = ply_file.split('/')[-1].split('.')[0].split('_')[0]
            part_id = ply_file.split('/')[-1].split('.')[0].split('_')[1]

            # read point cloud
            pcd = open3d.io.read_point_cloud(ply_file)

            # downsample
            fps_samples_idx = bucket_fps_kdtree_sampling(np.array(pcd.points), N_point)
            downsampled_pcd = open3d.geometry.PointCloud()
            downsampled_pcd.points = open3d.utility.Vector3dVector(np.array(pcd.points)[fps_samples_idx])

            # save
            save_path = os.path.join(save_root, f'{object_id}_{part_id}.ply')
            open3d.io.write_point_cloud(save_path, downsampled_pcd)

def gen_pcd_specified(pcd_path, save_path):
    pcd = open3d.io.read_point_cloud(pcd_path)
    N_point = 10000
    fps_samples_idx = bucket_fps_kdtree_sampling(np.array(pcd.points), N_point)
    downsampled_pcd = open3d.geometry.PointCloud()
    downsampled_pcd.points = open3d.utility.Vector3dVector(np.array(pcd.points)[fps_samples_idx])
    open3d.io.write_point_cloud(save_path, downsampled_pcd)

if __name__ == '__main__':
    paths = ['19179.ply', '19898.ply']
    for file in paths:
        gen_pcd_specified(file, file.replace('.ply', '_fps.ply'))



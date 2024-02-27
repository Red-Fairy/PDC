import os, sys, glob

import numpy as np

ori_root = "/data/haoran/Projects/GAPartNet_docs/part_meshes"
# /round_fixed_handle/round_fixed_handle_8983_2.npy
pose_paths = glob.glob(ori_root + "/*/*.npy")
mesh_paths = glob.glob(ori_root + "/*/*.obj")
print(len(pose_paths), len(mesh_paths))
import pdb; pdb.set_trace()
for pose_path in pose_paths:
    data = np.load(pose_path, allow_pickle=True).item()
    
    link_bbox = data['link_bbox']
    import open3d as o3d
    point_cloud = np.array(link_bbox)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:,:3])
    o3d.io.write_point_cloud(f"output_test.ply", pcd)
        
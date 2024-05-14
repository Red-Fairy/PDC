import h5py
import numpy as np
import os
import skimage
import trimesh
from pytorch3d.loss import chamfer_distance
import open3d
import torch
from tqdm import tqdm

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def visual_and_save_sdf(sdf, level=0.02, res=128):
    vertices, faces, normals, _ = skimage.measure.marching_cubes(sdf, level=level, spacing=(2/res, 2/res, 2/res))
    mesh_recon = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
    mesh_recon.export(f'debug/debug-{level}.obj')
    # print the bounding box of the mesh
    print('level: ', level)
    print('bounding box: ', mesh_recon.bounding_box.extents)

def sdf_to_o3d_mesh(sdf, level=0.02, res=128):
    vertices, faces, normals, _ = skimage.measure.marching_cubes(sdf, level=level, spacing=(2/res, 2/res, 2/res))
    mesh_recon = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
    mesh_o3d = open3d.geometry.TriangleMesh()
    mesh_o3d.vertices = open3d.utility.Vector3dVector(mesh_recon.vertices)
    mesh_o3d.triangles = open3d.utility.Vector3iVector(mesh_recon.faces)
    mesh_o3d.compute_vertex_normals()
    return mesh_o3d

root = '/mnt/data-rundong/PartDiffusion/dataset/part_sdf_128/slider_drawer'

avg_meters = {f'{i*0.0025:.4f}': AverageMeter() for i in range(4, 9)}

files = os.listdir(root)

max_failed_level = 0

for sdf_h5_file in tqdm(files):

    sdf_h5_file = os.path.join(root, sdf_h5_file)

    h5_f = h5py.File(sdf_h5_file, 'r')
    res = 128
    sdf = h5_f['pc_sdf_sample'][:].astype(np.float32).reshape(res, res, res)
    
    mesh_gt = open3d.io.read_triangle_mesh(sdf_h5_file.replace('part_sdf_128', 'part_meshes').replace('.h5', '.obj'))
    mesh_gt.translate(-mesh_gt.get_axis_aligned_bounding_box().get_center())
    pcd_gt = mesh_gt.sample_points_uniformly(number_of_points=10000)
    pcd_gt = torch.tensor(np.asarray(pcd_gt.points)).to('cuda').unsqueeze(0)

    for i in range(4, 9):
        try:
            mesh_recon = sdf_to_o3d_mesh(sdf, level=i*0.0025)
            mesh_recon.translate(-mesh_recon.get_axis_aligned_bounding_box().get_center())

            pcd_recon = mesh_recon.sample_points_uniformly(number_of_points=10000)
            pcd_recon = torch.tensor(np.asarray(pcd_recon.points)).to('cuda').unsqueeze(0)

            loss = chamfer_distance(pcd_recon, pcd_gt, batch_reduction='sum', point_reduction='mean')[0]
            avg_meters[f'{i*0.0025:.4f}'].update(loss.item())
        except:
            if i*0.0025 > max_failed_level:
                max_failed_level = i*0.0025
            print(f'{i*0.0025:.4f} error for {sdf_h5_file}')

# for key in avg_meters:
#     print(f'Level {key}: {avg_meters[key].avg}')

    # print(sdf.shape)

    # for i in range(1, 10):
    #     try:
    #         visual_and_save_sdf(sdf, level=i*0.0025)
    #     except:
    #         print(f'{i} error')

    # print(sdf.min())

    # # print the min and max value of the sdf
    # if sdf.min() < 0:
    #     print(sdf_h5_file)

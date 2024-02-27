import numpy as np
import h5py
import torch
import os
import skimage
import trimesh

h5_f = h5py.File('data/GAPartNet.v.0.1/handle/45146_1_merged_new_sdf_64.h5', 'r')

sdf = h5_f['pc_sdf_sample'][:].astype(np.float32)
sdf = torch.Tensor(sdf).view(64, 64, 64).cpu().numpy()
ori = h5_f['pc_sdf_original'][:].astype(np.float32)
ori = torch.Tensor(ori).view(1, 3).cpu().numpy()
print(f'sdf shape: {sdf.shape}')
print(f'ori shape: {ori.shape}')
print(f'ori value: {ori}')

vertices, faces, normals, _ = skimage.measure.marching_cubes(sdf, level=0, spacing=(0.01,0.01,0.01))

mesh_mar = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
# mesh_mar.show()
# mesh_mar.export('front3d_obj_tool/test/meshsdf.obj')
mar_bounding = mesh_mar.bounding_box
mar_cen = mesh_mar.bounding_box.centroid
new_vertices = mesh_mar.vertices - mar_cen
new_mesh = trimesh.Trimesh(new_vertices, mesh_mar.faces)

# marching_mesh_ex = mesh_mar.bounding_box.extents

# new_mesh.show()
new_mesh.export('/home/puhao/cache/partdiff/test.obj')
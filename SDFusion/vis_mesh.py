import h5py
import numpy as np
import trimesh
import skimage
import torch

sdf_h5_file = '/raid/haoran/Project/PartDiffusion/PartDiffusion/dataset/part_sdf_128/slider_drawer/12085_1.h5'
h5_f = h5py.File(sdf_h5_file, 'r')
sdf = h5_f['pc_sdf_sample'][:].astype(np.float32)
sdf = sdf.reshape(128, 128, 128)

SPACING = 2 / 128.
vertices, faces, normals, _ = skimage.measure.marching_cubes(sdf, level=0.02, spacing=(SPACING, SPACING, SPACING))
mesh_recon = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)

mesh_recon.apply_translation(-mesh_recon.bounding_box.centroid)
# export
recon_filename = 'debug.obj'
mesh_recon.export(recon_filename)
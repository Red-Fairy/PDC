import sys
sys.path.append('.')
from utils.util_3d import sdf_to_mesh_trimesh, mesh_to_sdf
from tqdm import tqdm
import os
import numpy as np
import h5py
import trimesh

# root = '/mnt/data-rundong/PartDiffusion/dataset/part_sdf_128/'
src_root = '/mnt/data-rundong/PartDiffusion/dataset/full_meshes_manifold'
dst_root = '../../part_meshes_recon'

cats = ['slider_drawer', 'hinge_door', 'hinge_knob', 'line_fixed_handle']
align_methods = ['volume', 'max_extent', 'volume', 'volume']

res = 128
level = 0.02

for cat, align_method in zip(cats, align_methods):
    files = [x for x in os.listdir(os.path.join(src_root, cat)) if x.endswith('.obj')]
    for file in tqdm(files):
        mesh = trimesh.load(os.path.join(src_root, cat, file))
        centroid, extents = mesh.bounding_box.centroid, mesh.extents

        sdf = mesh_to_sdf(mesh, res, padding=0.2, trunc=0.2).detach().cpu().numpy().reshape(res, res, res)
        # filepath = os.path.join(root, cat, file)
        # if not file.endswith('.h5'):
        #     continue
        # h5_f = h5py.File(filepath, 'r')
        # sdf = h5_f['pc_sdf_sample'][:].astype(np.float32).reshape(res, res, res)
        # h5_f.close()
        
        mesh_recon = sdf_to_mesh_trimesh(sdf, level=0.02, spacing=(2./res, 2./res, 2./res))
        if align_method == 'volume':
            scale = np.prod(extents)**(1/3) / np.prod(mesh_recon.extents)**(1/3)
        else:
            scale = np.max(extents) / np.max(mesh_recon.extents)
        mesh_recon.apply_scale(scale)
        mesh_recon.apply_translation(centroid) # move back to original position

        save_path = os.path.join(dst_root, cat, file.replace('.h5', '.obj'))

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        mesh_recon.export(save_path)
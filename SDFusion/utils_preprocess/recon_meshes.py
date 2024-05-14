import sys
sys.path.append('.')
from utils.util_3d import sdf_to_mesh_trimesh
from tqdm import tqdm
import os
import numpy as np
import h5py

root = '/mnt/data-rundong/PartDiffusion/dataset/part_sdf_128/'
save_root = '/mnt/azureml/cr/j/18805bb336d249f6b409578dd210054f/exe/wd/part_meshes_recon'

cats = ['slider_drawer', 'hinge_door', 'hinge_knob', 'line_fixed_handle']

res = 128
level = 0.02

for cat in cats:
    files = os.listdir(os.path.join(root, cat))
    for file in tqdm(files):
        if not file.endswith('.h5'):
            continue
        h5_f = h5py.File(file, 'r')
        sdf = h5_f['pc_sdf_sample'][:].astype(np.float32).reshape(res, res, res)
        h5_f.close()
        
        mesh_recon = sdf_to_mesh_trimesh(sdf, level=0.02, spacing=(2./res, 2./res, 2./res))
        save_path = os.path.join(save_root, cat, file.replace('.h5', '.obj'))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        mesh_recon.export(save_path)
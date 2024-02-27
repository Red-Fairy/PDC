from utils.util_3d import sdf_to_mesh_trimesh, mesh_to_sdf
import os
import h5py
import numpy as np
import trimesh
from termcolor import colored, cprint


if __name__ == '__main__':
    CAT = 'slider_drawer'
    metadata_basedir = f'/mnt/seagate12t/GAPartNet/v2/part_meshes_wbbox/{CAT}'
    mesh_basedir = f'/mnt/seagate12t/GAPartNet/v2/part_meshes_manifold/{CAT}'
    saved_mesh_basedir = f'/mnt/seagate12t/GAPartNet/v2_cen_boundscale1.2/part_meshes_manifold/{CAT}'
    if not os.path.exists(saved_mesh_basedir):
        os.makedirs(saved_mesh_basedir)

    metadata = {}
    for file_name in os.listdir(metadata_basedir):
        if not file_name.endswith('.npy'):
            continue
        file_path = os.path.join(metadata_basedir, file_name)
        metadata[file_name[:-4]] = np.load(file_path, allow_pickle=True).item()
    
    ## load mesh 
    for file_name in os.listdir(mesh_basedir):
        if not file_name.endswith('.obj'):
            continue
        if file_name[:-6] not in metadata.keys():
            cprint(f'file_name: {file_name} not in metadata', 'red')
            continue
        cprint(f'process fuxking mesh: {file_name}', 'green')
        mesh_path = os.path.join(mesh_basedir, file_name)
        mesh = trimesh.load(mesh_path)
        
        ## center the mesh
        mesh.apply_translation(-mesh.centroid)
        ## scale the mesh
        mesh_xlength = np.max(mesh.vertices[:, 0]) - np.min(mesh.vertices[:, 0])
        if mesh_xlength > 1.2:
            mesh_scale = 1.2 / mesh_xlength
            mesh.apply_scale(mesh_scale)
        
        ## save mesh
        saved_mesh_path = os.path.join(saved_mesh_basedir, file_name[:-6] + '.obj')
        mesh.export(saved_mesh_path)

    


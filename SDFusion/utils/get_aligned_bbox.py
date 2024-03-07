import trimesh
import os
import numpy as np
import json
import open3d

cat = 'slider_drawer'
root = f'/raid/haoran/Project/PartDiffusion/PartDiffusion/dataset/part_meshes/{cat}/'
save_root = f'/raid/haoran/Project/PartDiffusion/PartDiffusion/dataset/part_bbox_alingned/{cat}'
os.makedirs(save_root, exist_ok=True)
meshes = sorted(os.listdir(root))
meshes = [os.path.join(root, mesh) for mesh in meshes if mesh.endswith('.obj')]

for mesh_path in meshes:

    id = mesh_path.split('/')[-1].split('.')[0].split('_')[-2]
    part_id =  mesh_path.split('/')[-1].split('.')[0].split('_')[-1]

    mesh = trimesh.load(mesh_path)
    centroid = mesh.bounding_box.centroid
    extents = mesh.extents

    # save into json
    bbox = {
        'centroid': centroid.tolist(),
        'extents': extents.tolist()
    }
    with open(os.path.join(save_root, f'{id}_{part_id}.json'), 'w') as f:
        json.dump(bbox, f)
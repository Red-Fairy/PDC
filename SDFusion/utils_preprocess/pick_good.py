import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('root', type=str, default='./logs/slider_drawer-ply2shape-plyrot-scale3-lr0.00001/test_250000_rotate0.0_scale3.0_eta0.0_steps50_volume_mobility_diversity_margin256-haoran')
args = parser.parse_args()

root = args.root

with open(os.path.join(root, 'log.txt'), 'r') as f:
    lines = f.readlines()

lines = [line for line in lines if 'best' in line]

dst_root_mesh = os.path.join(root, 'meshes_selected')
dst_root_mesh_canonical = os.path.join(root, 'meshes_canonical_selected')
dst_root_pcd = os.path.join(root, 'pcd_selected')
os.makedirs(dst_root_mesh, exist_ok=True)
os.makedirs(dst_root_pcd, exist_ok=True)
os.makedirs(dst_root_mesh_canonical, exist_ok=True)

for line in tqdm(lines):
    object_id, part_id = line.split(' ')[1].split('_')
    instance_id = int(line.split(' ')[3][:-1])
    src = os.path.join(root, 'pcd', f'{object_id}_{part_id}_{instance_id}.ply')
    dst = os.path.join(dst_root_pcd, f'{object_id}_{part_id}.ply')
    os.system(f'cp {src} {dst}')
    src = os.path.join(root, 'meshes_canonical', f'{object_id}_{part_id}_{instance_id}.obj')
    dst = os.path.join(dst_root_mesh_canonical, f'{object_id}_{part_id}.obj')
    os.system(f'cp {src} {dst}')
    src = os.path.join(root, 'meshes', f'{object_id}_{part_id}_{instance_id}.obj')
    dst = os.path.join(dst_root_mesh, f'{object_id}_{part_id}.obj')
    os.system(f'cp {src} {dst}')


    
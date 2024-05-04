import os
from tqdm import tqdm

root = '/mnt/azureml/cr/j/19c62471467141d39f5f0dc988c1ea42/exe/wd/PartDiffusion/SDFusion/logs/slider-ply2shape-plyrot-scale3-lr0.00001/test_250000_rotate0.0_scale3.0_eta0.0_steps50_volume_mobility_diversity_margin128-haoran'


with open(os.path.join(root, 'log.txt'), 'r') as f:
    lines = f.readlines()

lines = [line for line in lines if 'best' in line]

dst_root = os.path.join(root, 'selected')
os.makedirs(dst_root, exist_ok=True)

for line in tqdm(lines):
    object_id, part_id = line.split(' ')[1].split('_')
    instance_id = int(line.split(' ')[3][:-1])
    src = os.path.join(root, f'{object_id}_{part_id}-{instance_id}.ply')
    dst = os.path.join(dst_root, f'{object_id}_{part_id}.ply')
    os.system(f'cp {src} {dst}')
    src = os.path.join(root, f'{object_id}_{part_id}-{instance_id}.obj')
    dst = os.path.join(dst_root, f'{object_id}_{part_id}.obj')
    os.system(f'cp {src} {dst}')


    
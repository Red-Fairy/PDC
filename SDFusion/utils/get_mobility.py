import trimesh
import os
import numpy as np
import json
import open3d
from tqdm import tqdm

cat = 'slider_drawer'

root = f'/raid/haoran/Project/PartDiffusion/PartDiffusion/dataset/part_ply_fps/{cat}/'

save_root = f'/raid/haoran/Project/PartDiffusion/PartDiffusion/dataset/part_mobility/{cat}'

annotation_root = '/raid/haoran/Project/data/partnet_all_annotated_new/annotation/'

os.makedirs(save_root, exist_ok=True)

obj_names = sorted(os.listdir(root))

for obj_name in tqdm(obj_names):

    id = obj_name.split('/')[-1].split('.')[0].split('_')[-2]
    part_id =  obj_name.split('/')[-1].split('.')[0].split('_')[-1]

    print(id, part_id)

    annotation_json = os.path.join(annotation_root, id, 'mobility_v2.json')

    json_file = json.load(open(annotation_json))

    for obj in json_file:
        if obj['id'] == int(part_id):
            mobility_dict = {}
            joint_data = obj['jointData']
            move_axis = joint_data['axis']['direction']
            move_origin = joint_data['axis']['origin']
            move_limit = [joint_data['limit']['a'], joint_data['limit']['b']]

            mobility_dict['move_axis'] = move_axis
            mobility_dict['move_origin'] = move_origin
            mobility_dict['move_limit'] = move_limit

            with open(os.path.join(save_root, f'{id}_{part_id}.json'), 'w') as f:
                json.dump(mobility_dict, f)

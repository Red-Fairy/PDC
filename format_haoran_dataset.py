import os
import open3d
from scipy.spatial.transform import Rotation as R
import numpy as np
import json
from tqdm import tqdm

data_count_per_instance = 3

root = './haoran'
files = os.listdir(root)
files = [file for file in files if 'pred_bbox' in file]

file_count_record = {}

for i in range(data_count_per_instance):
    os.makedirs(f'./ignore_files/instance_{i}', exist_ok=True)

for file in tqdm(files):
    # obtain the object_id, part_id, rotate angle (inferred from filename)
    # and the bbox (inferred from the bbox file)
    object_id = int(file.split('-')[0])
    part_id = int(file.split('link_')[1].split('-')[0])
    rotate_angle = float(file.split('-pred_bbox')[0].split('_')[-1])
    key = (object_id, part_id)
    if key not in file_count_record:
        file_count_record[key] = 0
    else:
        file_count_record[key] += 1
    dst_path = f'./ignore_files/instance_{file_count_record[key]}/{object_id}_{part_id}.json'

    points = open3d.io.read_point_cloud(os.path.join(root, file)).points
    points = np.array(points)
    # due to different camera conventions, rotate by zyx_ruler by (-90, 0, 90), 
    # then by xyz_rule by (0, -rotate_angle, 0)
    r = R.from_euler('zyx', [90, 0, -90], degrees=True)
    points = points @ r.as_matrix().T
    r = R.from_euler('xyz', [0, -rotate_angle, 0], degrees=True)
    points = points @ r.as_matrix().T

    # translation part is equal to the bbox center
    translation = points.mean(axis=0)
    # get the extent of the bbox
    extent = points.max(axis=0) - points.min(axis=0)
    bbox = {
        'centroid': translation.tolist(),
        'extents': extent.tolist(),
        'rotate_angle': rotate_angle
    }
    # save json
    with open(dst_path, 'w') as f:
        json.dump(bbox, f)



import os
import open3d
from scipy.spatial.transform import Rotation as R
import numpy as np
import json
from tqdm import tqdm
import torch

data_count_per_instance = 5

root = '/mnt/azureml/cr/j/7aaae224465249e79ffd6396abc43217/exe/wd/data-rundong/PartDiffusion/eval_output'
dst_root = '../ignore_files/slider_drawer'
files = os.listdir(root)
files = [file for file in files if 'pred_bbox' in file]

file_count_record = {}

sum_angle_diff = 0

for i in range(data_count_per_instance):
    os.makedirs(os.path.join(dst_root, f'set_{i}'), exist_ok=True)

for i, file in enumerate(tqdm(files)):
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
    dst_path = os.path.join(dst_root, f'set_{file_count_record[key]}/{object_id}_{part_id}.json')

    points = open3d.io.read_point_cloud(os.path.join(root, file)).points
    points = np.array(points)
    # due to different camera conventions, rotate by zyx_ruler by (-90, 0, 90), 
    # then calculate the rotation angle around the y-axis
    r = R.from_euler('zyx', [90, 0, -90], degrees=True)
    points = points @ r.as_matrix().T
    
    xyz_mean = points.mean(axis=0) # (3,)
    # print(xyz_mean)
    x_positive = (points[1]+points[2]+points[5]+points[6])/4
    # print(x_positive)
    x_positive_dir = x_positive - xyz_mean
    # print(x_positive_dir)

    # calculate the rotation angle around the y-axis
    rotate_angle_pred = 90 - np.arctan2(x_positive_dir[2], x_positive_dir[0]) * 180 / np.pi

    # rotate the points by the predicted angle
    r = R.from_euler('y', -rotate_angle_pred, degrees=True)
    points = points @ r.as_matrix().T

    # print(rotate_angle, rotate_angle_pred)
    sum_angle_diff += abs(rotate_angle - rotate_angle_pred)

    # r = R.from_euler('xyz', [0, -rotate_angle, 0], degrees=True)
    # points = points @ r.as_matrix().T

    # translation part is equal to the bbox center
    translation = points.mean(axis=0)
    # get the extent of the bbox
    extent = points.max(axis=0) - points.min(axis=0)
    bbox = {
        'centroid': translation.tolist(),
        'extents': extent.tolist(),
        'rotate_angle': rotate_angle,
        'rotate_angle_pred': rotate_angle_pred
    }
    # save json
    with open(dst_path, 'w') as f:
        json.dump(bbox, f)

print(sum_angle_diff / len(files))


def representation_to_bbox(representation):
    # representation is (B,12)
    # center is (B,3)
    center = representation[:,:3]
    # x is (B,3)
    x = representation[:,3:6]
    # y is (B,3)
    y = representation[:,6:9]
    # z is (B,3)
    z = representation[:,9:12]
    
    bbox = torch.zeros((representation.shape[0],8,3))
    bbox[:,0] = center - x/2 - y/2 - z/2
    bbox[:,1] = center + x/2 - y/2 - z/2
    bbox[:,2] = center + x/2 + y/2 - z/2
    bbox[:,3] = center - x/2 + y/2 - z/2
    bbox[:,4] = center - x/2 - y/2 + z/2
    bbox[:,5] = center + x/2 - y/2 + z/2
    bbox[:,6] = center + x/2 + y/2 + z/2
    bbox[:,7] = center - x/2 + y/2 + z/2
    return bbox

def bbox_to_representation(bbox):
    # bbox is (B,8,3)
    # center is (B,3)
    center = bbox.mean(dim=1)
    # -x -> +x
    x = bbox[:,1] - bbox[:,0]
    # -y -> +y
    y = bbox[:,3] - bbox[:,0]
    # -z -> +z
    z = bbox[:,4] - bbox[:,0]
    representation = torch.cat([center, x, y, z], dim=1).reshape(-1,12)
    
    return representation



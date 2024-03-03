import os
import sys
from os.path import join as pjoin
import numpy as np
from argparse import ArgumentParser
import json
from tqdm import tqdm
import trimesh

from config_utils import PARTNET_DATASET_PATH, AKB48_DATASET_PATH, PARTNET_ID_PATH, AKB48_ID_PATH, PARTNET_CAMERA_POSITION_RANGE, \
    AKB48_CAMERA_POSITION_RANGE, TARGET_GAPARTS, BACKGROUND_RGB, SAVE_PATH
from read_utils import get_id_category, read_joints_from_urdf_file
from render_utils import get_cam_pos, set_all_scene
from pose_utils import query_part_pose_from_joint_qpos, compute_rotation_matrix

def compute_all_transformation_matrix(link_names, link_pose_dict):
    NPCS_RTS_dict = {}
    for link_name in link_names:
        bbox = link_pose_dict[link_name]['bbox']
        T = bbox.mean(axis=0)
        s_x = np.linalg.norm(bbox[1] - bbox[0])
        s_y = np.linalg.norm(bbox[1] - bbox[2])
        s_z = np.linalg.norm(bbox[0] - bbox[4])
        S = np.array([s_x, s_y, s_z])
        scaler = np.linalg.norm(S)
        bbox_scaled = (bbox - T) / scaler
        bbox_canon = np.array([
            [-s_x / 2, s_y / 2, s_z / 2],
            [s_x / 2, s_y / 2, s_z / 2],
            [s_x / 2, -s_y / 2, s_z / 2],
            [-s_x / 2, -s_y / 2, s_z / 2],
            [-s_x / 2, s_y / 2, -s_z / 2],
            [s_x / 2, s_y / 2, -s_z / 2],
            [s_x / 2, -s_y / 2, -s_z / 2],
            [-s_x / 2, -s_y / 2, -s_z / 2]
        ]) / scaler
        R = compute_rotation_matrix(bbox_canon, bbox_scaled)
        NPCS_RTS_dict[link_name] = {'R': R, 'T': T, 'S': S, 'scaler': scaler}
    
    return NPCS_RTS_dict

def main(dataset_name, model_id, camera_idx, height, width, wanted_cat='slider_drawer'):
    
    # 1. read the id list to get the category; set path, camera range, and base link name
    if dataset_name == 'partnet':
        category = get_id_category(model_id, PARTNET_ID_PATH)
        if category is None:
            raise ValueError(f'Cannot find the category of model {model_id}')
        data_path = pjoin(PARTNET_DATASET_PATH, str(model_id))
        camera_position_range = PARTNET_CAMERA_POSITION_RANGE
        base_link_name = 'base'
        
    elif dataset_name == 'akb48':
        category = get_id_category(model_id, AKB48_ID_PATH)
        if category is None:
            raise ValueError(f'Cannot find the category of model {model_id}')
        data_path = pjoin(AKB48_DATASET_PATH, category, str(model_id))
        camera_position_range = AKB48_CAMERA_POSITION_RANGE
        base_link_name = 'root'
    
    else:
        raise ValueError(f'Unknown dataset {dataset_name}')

    link_names = []
    cat_exist = False
    anno_file = pjoin(data_path, 'link_anno_gapartnet.json')
    anno_list = json.load(open(anno_file, 'r'))
    for link_dict in anno_list:
        if not link_dict['is_gapart']:
            continue
        if link_dict['category'] == wanted_cat:
            link_names.append(link_dict['link_name'])
            cat_exist = True
    if not cat_exist:
        return
    
    json_path = pjoin(data_path, 'link_anno_gapartnet.json')
    link_anno = json.load(open(json_path, 'r'))
    link_pose_dict = {}
    for link_dict in link_anno:
        if link_dict['link_name'] in link_names:
            link_pose_dict[link_dict['link_name']] = link_dict
    
    part_mesh_root = '/raid/haoran/Project/PartDiffusion/PartDiffusion/dataset/part_meshes'
    # get the extent of the mesh
    for link_name in link_names:
        link_id = link_name.split('_')[-1]
        part_mesh = pjoin(part_mesh_root, wanted_cat, f'{wanted_cat}_{model_id}_{link_id}.obj')
        mesh = trimesh.load_mesh(part_mesh)
        s_x, s_y, s_z = mesh.extents
        bbox_canon = np.array([
            [-s_x / 2, s_y / 2, s_z / 2],
            [s_x / 2, s_y / 2, s_z / 2],
            [s_x / 2, -s_y / 2, s_z / 2],
            [-s_x / 2, -s_y / 2, s_z / 2],
            [-s_x / 2, s_y / 2, -s_z / 2],
            [s_x / 2, s_y / 2, -s_z / 2],
            [s_x / 2, -s_y / 2, -s_z / 2],
            [-s_x / 2, -s_y / 2, -s_z / 2]
        ])
        bbox_transformed = np.array(link_pose_dict[link_name]['bbox'])
        T = bbox_transformed.mean(axis=0)
        bbox_transformed -= T
        # compute the rotation between vector bbox_canon[0] and bbox_transformed[0]
        M = np.dot(bbox_canon.T, bbox_transformed)
        U, S, Vt = np.linalg.svd(M)
        R = np.dot(Vt.T, U.T)
        if np.linalg.det(R) < 0:
            Vt[2] *= -1
            R = np.dot(Vt.T, U.T)
        # test the rotation matrix
        print(model_id)
        print(bbox_canon @ R, bbox_transformed)
        print(bbox_canon, bbox_transformed, np.round(R, 4))
        exit()
        # R = compute_rotation_matrix(bbox_canon, bbox_transformed)
        # # if R is not identity matrix, then print model_id and link_id
        R_abs = np.abs(R)
        if not np.allclose(np.round(R_abs,4), np.eye(3)):
            print(f"model_id: {model_id}, link_id: {link_id}")
        # if model_id == 102194:
        #     print(bbox_canon, bbox_transformed)

        save_root = '/raid/haoran/Project/PartDiffusion/PartDiffusion/dataset/part_transformation'
        save_path = pjoin(save_root, wanted_cat, f'{wanted_cat}_{model_id}_{link_id}.json')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            # convert numpy array to list, and keep 4 decimal places
            R = np.round(R, 4).tolist()
            T = np.round(T, 4).tolist()
            json.dump({'R': R, 'T': T}, f)
        
    
    # # 2. read the urdf file,  get the kinematic chain, and collect all the joints information
    # joints_dict = read_joints_from_urdf_file(data_path, 'mobility_relabel_gapartnet.urdf')
    
    # # 3. generate the joint qpos randomly in the limit range
    # joint_qpos = {}
    # for joint_name in joints_dict:
    #     joint_type = joints_dict[joint_name]['type']
    #     if joint_type == 'prismatic' or joint_type == 'revolute':
    #         joint_limit = joints_dict[joint_name]['limit']
    #         joint_qpos[joint_name] = np.random.uniform(joint_limit[0], joint_limit[1])
    #     elif joint_type == 'fixed':
    #         joint_qpos[joint_name] = 0.0  # ! the qpos of fixed joint must be 0.0
    #     elif joint_type == 'continuous':
    #         joint_qpos[joint_name] = np.random.uniform(-10000.0, 10000.0)
    #     else:
    #         raise ValueError(f'Unknown joint type {joint_type}')
    
    # # 4. generate the camera pose randomly in the specified range
    # camera_range = camera_position_range[category][camera_idx]
    # camera_pos = get_cam_pos(
    #     theta_min=camera_range['theta_min'], theta_max=camera_range['theta_max'],
    #     phi_min=camera_range['phi_min'], phi_max=camera_range['phi_max'],
    #     dis_min=camera_range['distance_min'], dis_max=camera_range['distance_max']
    # )
    
    # # 5. pass the joint qpos and the augmentation parameters to set up render environment and robot
    # scene, camera, engine, robot = set_all_scene(data_path=data_path, 
    #                                     urdf_file='mobility_relabel_gapartnet.urdf',
    #                                     cam_pos=camera_pos,
    #                                     width=width, 
    #                                     height=height,
    #                                     use_raytracing=False,
    #                                     joint_qpos_dict=joint_qpos)
    
    # # 6. use qpos to calculate the gapart poses
    # tgt_part_cat = [wanted_cat]
    # link_pose_dict = query_part_pose_from_joint_qpos(data_path=data_path, anno_file='link_anno_gapartnet.json', joint_qpos=joint_qpos, joints_dict=joints_dict, target_parts=tgt_part_cat, base_link_name=base_link_name, robot=robot)

    # all_transformation = compute_all_transformation_matrix(link_names, link_pose_dict)

    # save_root = os.path.join('/raid/haoran/Project/PartDiffusion/PartDiffusion/dataset/part_transformation', wanted_cat)
    # os.makedirs(save_root, exist_ok=True)
    
    # for link_name, transformation in all_transformation.items():
    #     # save as json file
    #     link_id = link_name.split('_')[-1]
    #     save_path = pjoin(save_root, f'{model_id}_{link_id}.json')
    #     with open(save_path, 'w') as f:
    #         # convert numpy array to list, and keep 4 decimal places
    #         transformation['R'] = np.round(transformation['R'], 4).tolist()
    #         transformation['T'] = np.round(transformation['T'], 4).tolist()
    #         transformation['S'] = np.round(transformation['S'], 4).tolist()
    #         transformation['scaler'] = round(transformation['scaler'], 4)
    #         json.dump(transformation, f)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='partnet', help='Specify the dataset to render')
    # parser.add_argument('--model_id', type=int, default=41083, help='Specify the model id to render')
    parser.add_argument('--camera_idx', type=int, default=0, help='Specify the camera range index to render')
    parser.add_argument('--height', type=int, default=800, help='Specify the height of the rendered image')
    parser.add_argument('--width', type=int, default=800, help='Specify the width of the rendered image')
    parser.add_argument('--wanted_cat', type=str, default='slider_drawer', help='Specify the category to render')

    args = parser.parse_args()
    
    assert args.dataset in ['partnet', 'akb48'], f'Unknown dataset {args.dataset}'
    if args.dataset == 'akb48':
        assert not args.replace_texture, 'Texture replacement is not needed for AKB48 dataset'

    model_ids = [int(id) for id in os.listdir('/raid/haoran/Project/data/partnet_all_annotated_new/annotation')]
    model_ids.sort()
    model_ids = [id for id in model_ids if id > 12085]

    for model_id in tqdm(model_ids):
        main(args.dataset, model_id, args.camera_idx, args.height, args.width, args.wanted_cat)
    
    print("Done!")

import glob
import numpy as np
import torch
import json, os
import xml.etree.ElementTree as ET
from os.path import join as pjoin
import pybullet as p

# Notice
# 1. visual meshes are exactly the same as collision meshes
# 2. meshes in a single link have exactly the same xyz coordinates
# 3. the bbox is in the object coordinate system

def combine_meshes(input_files, output_file):
    """
    Combine multiple .obj mesh files into a single file.
    
    Parameters:
    - input_files (list of str): List of input file paths.
    - output_file (str): Output file path.
    """
    vertices_list = []
    faces_list = []
    last_vertex_index = 0
    
    for file_name in input_files:
        with open(file_name, "r") as f:
            vertices = []
            faces = []
            for line in f:
                parts = line.strip().split()
                if len(parts) == 0:
                    continue
                if parts[0] == "v":
                    vertices.append(list(map(float, parts[1:])))
                elif parts[0] == "f":
                    # Update the vertex indices in face definitions
                    updated_face = [str(int(p.split('/')[0]) + last_vertex_index) for p in parts[1:]]
                    faces.append(updated_face)
            
            last_vertex_index += len(vertices)
            vertices_list.extend(vertices)
            faces_list.extend(faces)
    
    # Save combined mesh to output file
    with open(output_file, "w") as f:
        # Write vertices
        for vertex in vertices_list:
            f.write("v " + " ".join(map(str, vertex)) + "\n")
        
        # Write faces
        for face in faces_list:
            f.write("f " + " ".join(map(str, face)) + "\n")

def query_part_pose_from_joint_qpos(data_path, 
                                    anno_file, 
                                    joint_qpos, 
                                    joints_dict, 
                                    target_parts,
                                    remove_link = None):
    anno_path = pjoin(data_path, anno_file)
    anno_list = json.load(open(anno_path, 'r'))
    
    target_links = {}
    for link_dict in anno_list:
        link_name = link_dict['link_name']
        if remove_link is not None:
            if link_name == remove_link:
                continue
        is_gapart = link_dict['is_gapart']
        part_class = link_dict['category']
        bbox = link_dict['bbox']
        if is_gapart and part_class in target_parts:
            target_links[link_name] = {
                'category_id': target_parts.index(part_class),
                'bbox': np.array(bbox, dtype=np.float32).reshape(-1, 3)
            }
    
    joint_states = {}
    for joint in robot.get_joints():
        joint_name = joint.get_name()
        if joint_name in joints_dict:
            joint_pose = joint.get_parent_link().pose * joint.get_pose_in_parent()
            joint_states[joint_name] = {
                'origin': joint_pose.p,
                'axis': joint_pose.to_transformation_matrix()[:3,:3] @ [1,0,0]
            }
    
    child_link_to_joint_name = {}
    for joint_name, joint_dict in joints_dict.items():
        child_link_to_joint_name[joint_dict['child']] = joint_name
    
    result_dict = {}
    
    for link_name, link_dict in target_links.items():
        joint_names_to_base = []
        cur_name = link_name
        while cur_name in child_link_to_joint_name:
            joint_name = child_link_to_joint_name[cur_name]
            joint_names_to_base.append(joint_name)
            cur_name = joints_dict[joint_name]['parent']
        if not cur_name == 'base':
            continue
            import pdb; pdb.set_trace()
        joint_names_to_base = joint_names_to_base[:-1] # remove the last joint to 'base'
        
        bbox = link_dict['bbox']
        part_class = link_dict['category_id']
        for joint_name in joint_names_to_base[::-1]:
            joint_type = joints_dict[joint_name]['type']
            origin = joint_states[joint_name]['origin']
            axis = joint_states[joint_name]['axis']
            axis = axis / np.linalg.norm(axis)
            if joint_type == "fixed":
                continue
            elif joint_type == "prismatic":
                bbox = bbox + axis * joint_qpos[joint_name]
            elif joint_type == "revolute" or joint_type == "continuous":
                rotation_mat = t.axangle2mat(axis.reshape(-1).tolist(), joint_qpos[joint_name]).T
                bbox = np.dot(bbox - origin, rotation_mat) + origin
        
        result_dict[link_name] = {
            'category_id': part_class,
            'bbox': bbox
        }
    
    return result_dict

def get_link_index(robot_id, link_name):
    """
    Get the link index based on the link name.
    
    Parameters:
    - robot_id: int, The ID of the robot as returned by pybullet.loadURDF.
    - link_name: str, The name of the link.
    
    Returns:
    - int, The link index if found, otherwise -1.
    """
    num_joints = p.getNumJoints(robot_id)
    
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        if joint_info[12].decode("UTF-8") == link_name:  # joint_info[12] contains the joint name
            return i
    return -1

def get_norm_bbox(urdf_path, bbox_world, link_name):
    # Load the URDF and start simulation
    p.connect(p.DIRECT)  # Or p.GUI if you want a GUI
    print(urdf_path)
    try:
        robotID = p.loadURDF(urdf_path)
    except:
        p.disconnect()  
        return None
    p.setGravity(0, 0, 0.0)
    link_index = get_link_index(robotID, link_name)

    if link_index == -1:
        print(f"No link named {link_name} found!")
    else:

        # Assume you know the link index
        link_index = 0  # Replace with the actual index

        # Get the link state (position and orientation)
        link_state = p.getLinkState(robotID, link_index)
        link_pos, link_ori = link_state[0], link_state[1]

        # Inverse Transformation Matrix: [Rotation | Translation]
        rotation_matrix = np.array(p.getMatrixFromQuaternion(link_ori)).reshape(3, 3)
        translation_vector = np.array(link_pos)

        # import pdb; pdb.set_trace()
        # Calculate the bounding box coordinates in the link space
        bbox_world = np.array(bbox_world).reshape(-1, 3)
        bbox_link = np.dot(np.linalg.inv(rotation_matrix), (bbox_world - translation_vector.reshape(-1, 3)).T).T
        # bbox_max_link = np.dot(np.linalg.inv(rotation_matrix), (bbox_max_world - translation_vector))

        

    # Disconnect from the simulation
    p.disconnect()  
    return bbox_link 

data_root = "../data/partnet_all_annotated_new/annotation"
save_root = "/data/haoran/Projects/GAPartNet_docs/part_meshes"
# data/partnet_all_annotated_new/annotation/4108/link_anno_gapartnet.json
paths = glob.glob(data_root + "/*/link_anno_gapartnet.json")
total = len(paths)
i = 0
fails = []
for path in paths:
    i+=1
    print(i, total)
    model_id = path.split("/")[-2]
    print("model id:", model_id)
    f = open(path, "r").read()
    data = json.loads(f)
    num_link = len(data)
    # /data/haoran/Projects/GAPartNet_docs/data/partnet_all_annotated_new/annotation/4108/mobility_relabel_gapartnet.urdf
    urdf_path = f"../data/partnet_all_annotated_new/annotation/{model_id}/mobility_relabel_gapartnet.urdf"
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    links = [child for child in root if child.tag == "link" and child.attrib["name"] != "base"]
    if len(links) != num_link:
        import pdb; pdb.set_trace()
    for link_i in range(num_link):

        link_data = data[link_i]
        if not link_data["is_gapart"]:
            continue
        link_name = link_data["link_name"]
        link_category = link_data["category"]
        link_bbox = link_data["bbox"]
        link_info = links[link_i]
        visual_meshes = [f"../data/partnet_all_annotated_new/annotation/{model_id}/" + x.attrib['filename'] for x in link_info.findall("visual/geometry/mesh")]

        if os.path.exists(f"{save_root}/{link_category}/{link_category}_{model_id}_{link_i}.npy"):
            continue
        output_json_file = f"{save_root}/{link_category}/{link_category}_{model_id}_{link_i}.npy"
        os.makedirs(f"{save_root}/{link_category}", exist_ok=True)
        link_bbox_norm = get_norm_bbox(urdf_path, link_bbox, link_name)
        if link_bbox_norm is None:
            fails.append((model_id,link_i))
            continue
        data_save = {
            "model_id": model_id,
            "link_name": link_name,
            "link_bbox": link_bbox_norm,
            "category": link_category,
        }
        np.save(output_json_file, data_save, allow_pickle=True)
        print(i, total)
        # root.findall("joint")
        # for i in range(len(root.findall("joint"))):
        #     if root.findall("joint")[i][-1].attrib["link"] == "base":
        #         import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        # import open3d as o3d
        # point_cloud = np.array(link_bbox)
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(point_cloud[:,:3])
        # # pcd.colors = o3d.utility.Vector3dVector(point_cloud[:,3:6]/255)
        # # pcd.normals = o3d.utility.Vector3dVector(point_cloud[:,6:9])
        # o3d.io.write_point_cloud(f"output_link_{link_name}.ply", pcd)
        
        # output_file = f"{save_root}/{link_category}/{link_category}_{model_id}_{link_i}.obj"
        # combine_meshes(visual_meshes, output_file)
print(fails)
print(f"processing {len(paths)} models")
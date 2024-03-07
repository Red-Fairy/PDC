import json
import os
from tqdm import tqdm
import numpy as np

def find_parts_id(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    part_ids = []

    for obj in data:
        if obj['category'] == category:
            part_ids.append(int(obj['link_name'].split('_')[1]))

    return part_ids

def find_parts_and_children(json_file, part_ids):
    with open(json_file, 'r') as file:
        data = json.load(file)
        
    part_children_ids = []
    
    for obj in data:
        if obj['id'] in part_ids:
            part_id = obj['id']
            children_ids = [part['id'] for part in obj['parts']]
            part_children_ids.append((part_id, children_ids))
            
    return part_children_ids

# Function to read labels from the TXT file
def read_labels(txt_file):
    with open(txt_file, 'r') as file:
        labels = [int(line.strip()) for line in file]
    return labels

# Function to read points from the PLY file
def read_ply(ply_file):
    with open(ply_file, 'r') as file:
        header = True
        points = []
        while header:
            line = file.readline()
            if line.startswith("end_header"):
                header = False
            continue
        
        for line in file:
            if line.strip():
                points.append(list(map(float, line.strip().split())))
    return points

# Function to save points to a PLY file
def save_ply(points, file_name):
    with open(file_name, 'w') as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {len(points)}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("end_header\n")
        for point in points:
            file.write(f"{point[0]} {point[1]} {point[2]}\n")

def process_and_save_ply(part_children_ids, ply_file, labels_file, save_path):
    labels = read_labels(labels_file)
    points = read_ply(ply_file)
    
    for i, (part_id, children_ids) in enumerate(part_children_ids):
        # Filter points based on labels
        filtered_points = [point for j, point in enumerate(points) if labels[j] not in children_ids]
        
        # Save the filtered points to a new PLY file
        save_ply(filtered_points, f'{save_path}_{part_id}.ply')

def extract_and_save_bbox(json_file, part_id, save_path):
    with open(json_file, 'r') as file:
        data = json.load(file)
    for obj in data:
        if int(obj['link_name'].split('_')[1]) == part_id:
            bbox = obj['bbox']
            bbox_np = np.array(bbox).reshape(-1, 3)
            np.savetxt(f'{save_path}_{part_id}_bbox.txt', bbox_np, fmt='%f')
            break


root_path = '/raid/haoran/Project/data/partnet_all_annotated_new/annotation'

save_root_ply = '/raid/haoran/Project/PartDiffusion/PartDiffusion/dataset/part_ply'
save_root_bbox = '/raid/haoran/Project/PartDiffusion/PartDiffusion/dataset/part_bbox'

category = 'hinge_door'

os.makedirs(os.path.join(save_root_ply, category), exist_ok=True)
os.makedirs(os.path.join(save_root_bbox, category), exist_ok=True)

obj_ids = os.listdir(root_path)

for obj_id in tqdm(obj_ids):

    link_json_path = os.path.join(root_path, obj_id, 'link_anno_gapartnet.json')
    json_path = os.path.join(root_path, obj_id, 'mobility_v2.json')
    label_path = os.path.join(root_path, obj_id, 'point_sample/label-10000.txt')
    ply_path = os.path.join(root_path, obj_id, 'point_sample/ply-10000.ply')

    if not os.path.exists(link_json_path) or not os.path.exists(json_path) \
            or not os.path.exists(label_path) or not os.path.exists(ply_path):
        continue
            
    part_ids = find_parts_id(link_json_path)
    if len(part_ids) == 0:
        continue
    print(part_ids)

    part_children_ids = find_parts_and_children(json_path, part_ids)

    for part_id, children_ids in part_children_ids:
        print(f"Drawer ID: {part_id}, Children IDs: {children_ids}")
        extract_and_save_bbox(link_json_path, part_id, f'{save_root_bbox}/{category}/{obj_id}')

    process_and_save_ply(part_children_ids, ply_path, label_path, f'{save_root_ply}/{category}/{obj_id}')


import os, sys
import numpy as np
import glob 

def read_obj(filename):
    vertices = []
    faces = []
    
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 0:
                continue
            if parts[0] == 'v':
                vertices.append(tuple(map(float, parts[1:4])))
            elif parts[0] == 'f':
                # Extract the vertex indices from the face definition
                face_vertices = [int(p.split('/')[0]) for p in parts[1:]]
                faces.append(tuple(face_vertices))
    
    return vertices, faces

def write_obj(filename, vertices, faces):
    with open(filename, 'w') as file:
        for v in vertices:
            file.write("v {} {} {}\n".format(v[0], v[1], v[2]))
        for f in faces:
            file.write("f {} {} {}\n".format(f[0], f[1], f[2]))

def normalize_mesh(vertices):
    min_x = min(v[0] for v in vertices)
    max_x = max(v[0] for v in vertices)
    min_y = min(v[1] for v in vertices)
    max_y = max(v[1] for v in vertices)
    min_z = min(v[2] for v in vertices)
    max_z = max(v[2] for v in vertices)

    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    center_z = (min_z + max_z) / 2.0

    max_extent = max(max_x - min_x, max_y - min_y, max_z - min_z)
    scale = 1.0 / max_extent

    normalized_vertices = [((v[0] - center_x) * scale, 
                            (v[1] - center_y) * scale, 
                            (v[2] - center_z) * scale) for v in vertices]

    return normalized_vertices



part_names = ["handle", "button", "door", "drawer", "knob", "lid"]
for part_name in part_names:
    ROOT = "/data/haoran/data/SingleParts"
    paths = glob.glob(f"{ROOT}/{part_name}_merge/*.obj")
    SAVE_ROOT = f"/data/haoran/data/SingleParts/{part_name}_normalized"
    os.makedirs(SAVE_ROOT, exist_ok=True)

    for path in paths:
        print(path)
        vertices, faces = read_obj(path)
        normalized_vertices = normalize_mesh(vertices)
        name = path.split('/')[-1]
        
        save_path = os.path.join(SAVE_ROOT, name)
        write_obj(save_path, normalized_vertices, faces)
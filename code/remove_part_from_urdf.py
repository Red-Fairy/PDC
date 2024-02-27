import xml.etree.ElementTree as ET
from lxml import etree
import glob
import numpy as np
import torch
import json, os
import xml.etree.ElementTree as ET

# Notice
# 1. visual meshes are exactly the same as collision meshes
# 2. meshes in a single link have exactly the same xyz coordinates

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

# def remove_link_and_child(urdf_file, link_to_remove, save_path):
#     """
#     Remove a link and its child link from a URDF file.
    
#     Parameters:
#     - urdf_file (str): Path to the input URDF file.
#     - link_to_remove (str): Name of the link to remove.
#     """
#     parser = etree.XMLParser(remove_blank_text=True)  # Initialize the XML parser
#     tree = ET.parse(urdf_file, parser)  # Parse the XML file
#     root = tree.getroot()  # Get the root of the XML tree

#     # Find and remove the specified link
#     for link in root.findall('link'):
#         if link.attrib['name'] == link_to_remove:
#             root.remove(link)
#             break
    
#     # Find the joint that connects the specified link and remove it
#     for joint in root.findall('joint'):
#         if joint.find('parent').attrib['link'] == link_to_remove:
#             child_link_name = joint.find('child').attrib['link']
            
#             # Remove the joint
#             root.remove(joint)
            
#             # Remove the child link
#             for link in root.findall('link'):
#                 if link.attrib['name'] == child_link_name:
#                     root.remove(link)
#                     break
#             break
    
#     # Save the modified XML tree to a new URDF file
#     # tree.write(save_path)
#     new_tree = etree.ElementTree(root)  # Convert xml.etree.ElementTree.Element to lxml.etree._ElementTree
#     with open(save_path, 'wb') as f:  # Note 'wb' instead of 'w' to write bytes, not strings
#         new_tree.write(f, pretty_print=True, xml_declaration=True, encoding='utf-8')

# def remove_link_and_descendants(root, link_to_remove):
#     """
#     Recursively remove a link and all its descendant links from a URDF XML tree.
    
#     Parameters:
#     - root: Root element of the XML tree.
#     - link_to_remove (str): Name of the link to remove.
#     """
#     # Find and remove any joints and their child links that have the specified link as a parent
#     for joint in root.findall('joint'):
#         if joint.find('parent').attrib['link'] == link_to_remove:
#             child_link_name = joint.find('child').attrib['link']
            
#             # Recursively remove child link and its descendants
#             remove_link_and_descendants(root, child_link_name)
            
#             # Remove the joint
#             root.remove(joint)
    
#     # Remove the specified link
#     for link in root.findall('link'):
#         if link.attrib['name'] == link_to_remove:
#             root.remove(link)


# def remove_link_and_descendants(root, link_to_remove):
#     """
#     Recursively remove a link and all its descendant links from a URDF XML tree.
    
#     Parameters:
#     - root: Root element of the XML tree.
#     - link_to_remove (str): Name of the link to remove.
#     """
#     joints_to_remove = []
    
#     # Iterate through joints to find connections to remove
#     for joint in root.findall('joint'):
#         # Check if the link to remove is a parent in this joint
#         if joint.find('parent').attrib['link'] == link_to_remove:
#             child_link_name = joint.find('child').attrib['link']
#             joints_to_remove.append(joint)
#             remove_link_and_descendants(root, child_link_name)
#         # Check if the link to remove is a child in this joint
#         elif joint.find('child').attrib['link'] == link_to_remove:
#             joints_to_remove.append(joint)
    
#     # Remove the specified link
#     for link in root.findall('link'):
#         if link.attrib['name'] == link_to_remove:
#             root.remove(link)
    
#     # Remove the recorded joints
#     for joint in joints_to_remove:
#         root.remove(joint)

def remove_link_and_descendants(root, link_to_remove):
    joints_to_remove = []
    
    # Iterate through joints to find connections to remove
    for joint in root.findall('joint'):
        parent_link = joint.find('parent').attrib['link']
        child_link = joint.find('child').attrib['link']
        
        if parent_link == link_to_remove or child_link == link_to_remove:
            # joints_to_remove.append(joint)
            root.remove(joint)
            if parent_link == link_to_remove:
                remove_link_and_descendants(root, child_link)
                
    # Remove the specified link and joints directly under root
    for link in root.findall('link'):
        if link.attrib['name'] == link_to_remove:
            root.remove(link)
            
    # for joint in joints_to_remove:
    #     root.remove(joint)


def remove_link_and_child(urdf_file, link_to_remove, save_path):
    """
    Remove a link and all its descendant links from a URDF file.
    
    Parameters:
    - urdf_file (str): Path to the input URDF file.
    - link_to_remove (str): Name of the link to remove.
    """
    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(urdf_file, parser)
    root = tree.getroot()
    
    # Recursively remove the specified link and all its descendants
    remove_link_and_descendants(root, link_to_remove)

    # Save the modified XML tree to a new URDF file
    with open(save_path, 'wb') as f:
        tree.write(f, pretty_print=True, xml_declaration=True, encoding='utf-8')
  
data_root = "../data/partnet_all_annotated_new/annotation"
save_root = "/data/haoran/Projects/GAPartNet_docs/part_meshes"
# data/partnet_all_annotated_new/annotation/4108/link_anno_gapartnet.json
paths = glob.glob(data_root + "/*/link_anno_gapartnet.json")

for path in paths:
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
        save_path = "./test.urdf"
        print(save_path)
        # os.makedirs(f"../data/partnet_all_annotated_new/annotation/{model_id}", exist_ok=True)
        save_path = f"../data/partnet_all_annotated_new/annotation/{model_id}/remove_single_part_wo-{link_category}-{link_name}.urdf"
        # save_path = "test2.urdf"
        remove_link_and_child(urdf_path, link_name, save_path)
        # import pdb; pdb.set_trace()
        
        # visual_meshes = [f"../data/partnet_all_annotated_new/annotation/{model_id}/" + x.attrib['filename'] for x in link_info.findall("visual/geometry/mesh")]

        # output_file = f"{save_root}/{link_category}/{link_category}_{model_id}_{link_i}.obj"
        # os.makedirs(f"{save_root}/{link_category}", exist_ok=True)
        # combine_meshes(visual_meshes, output_file)
    
print(f"processing {len(paths)} models")

# Example Usage
# urdf_file = 'path_to_your_robot.urdf'
# link_to_remove = 'link_name_to_remove'
# remove_link_and_child(urdf_file, link_to_remove)

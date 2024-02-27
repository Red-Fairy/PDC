from utils.util_3d import sdf_to_mesh_trimesh
import h5py
import numpy as np
import trimesh
import glob

SDF_ROOT = "/raid/haoran/Project/data/part_sdf/slider_drawer"
MESH_INFO_ROOT = "/raid/haoran/Project/data/part_meshes/slider_drawer"
SAVE_ROOT = "/raid/haoran/Project/SDFusion/data_bbox"
LENGTH = 1.2
SPACING = LENGTH / 64  ## 1.2 / 64
def write_obj(points, file_name):
    with open(file_name, 'w') as f:
        for point in points:
            f.write(f"v {point[0]} {point[1]} {point[2]}\n")
        print("OBJ file created successfully!")

sdf_paths = glob.glob(SDF_ROOT + "/*.h5")

for sdf_path in sdf_paths:
    
    data_sdf = np.array(h5py.File(sdf_path, 'r')['pc_sdf_sample'])

    mesh = sdf_to_mesh_trimesh(data_sdf.reshape(64,64,64), spacing=(SPACING, SPACING, SPACING), level = 0.01)

    name = "_".join(sdf_path.split("/")[-1].split(".")[0].split("_")[:4])
    print(name)
    meta_path = f"{MESH_INFO_ROOT}/{name}.npy"

    mesh.export(f"{SAVE_ROOT}/{name}_sdf_mesh.obj", "obj")

    meta_data = np.load(meta_path, allow_pickle=True).item()
    link_bbox = meta_data["link_bbox"]
    mesh_bbox = trimesh.points.PointCloud(link_bbox)
    mesh_bbox.apply_translation(-mesh_bbox.centroid)
    mesh_bbox_xlength = np.max(mesh_bbox.vertices[:, 0]) - np.min(mesh_bbox.vertices[:, 0])
    if mesh_bbox_xlength > 1.2:
        mesh_bbox_scale = 1.2 / mesh_bbox_xlength
        mesh_bbox.apply_scale(mesh_bbox_scale)
    mesh_bbox.export(f"{SAVE_ROOT}/{name}_sdf_bbox.obj", "obj")
    mesh_bbox_array = np.array(mesh_bbox.vertices)
    np.save(f"{SAVE_ROOT}/{name}_sdf_bbox.npy", mesh_bbox_array)
    # import pdb; pdb.set_trace()

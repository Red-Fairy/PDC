import numpy as np
import torch
import h5py
import trimesh
import skimage

def sdf_to_mesh_trimesh(sdf, level=0.02,spacing=(0.01,0.01,0.01)):
    if torch.is_tensor(sdf):
        sdf = sdf.detach().cpu().numpy()
    vertices, faces, normals, _ = skimage.measure.marching_cubes(sdf, level=level, spacing=spacing)
    mesh_mar = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)

    mar_bounding = mesh_mar.bounding_box
    mar_cen = mesh_mar.bounding_box.centroid
    new_vertices = mesh_mar.vertices - mar_cen
    mesh = trimesh.Trimesh(new_vertices, mesh_mar.faces)

    return mesh

def calculate_rel_RT(sdf, bbox, scale=None, res=64, ret_mesh=False):
    '''
    Calculate the relative transformation between the canonical mesh specified by the sdf 
    and its target bounding box in the world space
    bbox in format [(-x,+y,+z), (+x,+y,+z), (+x,-y,+z), (-x,-y,+z), (-x,+y,-z), (+x,+y,-z), (+x,-y,-z), (-x,-y,-z)]
    '''
    # step 1, from sdf to mesh
    spc = (2./res, 2./res, 2./res)
    mesh = sdf_to_mesh_trimesh(sdf, level=0.02, spacing=spc)
    # move it to the center of the world
    mesh.vertices -= mesh.centroid
    # print(mesh.centroid)
    # scale it to the target size
    if scale is None: # use volume^1/3
        xyz2 = bbox[1] - bbox[7]
        print(xyz2)
        scale = (abs(xyz2[0] * xyz2[1] * xyz2[2])/ abs(mesh.extents[0] * mesh.extents[1] * mesh.extents[2])) ** (1/3)
    # print(scale)
    mesh.vertices *= scale
    print(mesh.extents)

    # step 2.1 calculate the translation
    translation = (bbox[1] + bbox[7]) / 2

    # step 2.2 calculate the rotation
    bounds = mesh.bounds
    print(f"bounds: {bounds}")
    x_mid, y_mid, z_mid = (bounds[0] + bounds[1]) / 2
    x_ori = np.array([bounds[1][0], y_mid, z_mid])
    y_ori = np.array([x_mid, bounds[1][1], z_mid])
    z_ori = np.array([x_mid, y_mid, bounds[1][2]])
    xyz_ori = np.stack([x_ori, y_ori, z_ori], axis=0) # (3, 3)
    print(xyz_ori)

    x_dst = np.array([(bbox[1][0] - bbox[0][0]) / 2, 0, 0])
    y_dst = np.array([0, (bbox[1][1] - bbox[2][1]) / 2, 0])
    z_dst = np.array([0, 0, (bbox[0][2] - bbox[4][2]) / 2])
    xyz_dst = np.stack([x_dst, y_dst, z_dst], axis=0) # (3, 3)
    print(xyz_dst)

    # R @ xyz_ori = xyz_dst
    R = np.linalg.solve(xyz_ori, xyz_dst)
    print(R)
    exit()

    RT = np.array([[Rx[0], Ry[0], Rz[0], translation[0]],
                     [Rx[1], Ry[1], Rz[1], translation[1]],
                     [Rx[2], Ry[2], Rz[2], translation[2]],
                     [0, 0, 0, 1]])
    if ret_mesh:
        return RT, mesh
    else:
        return RT

if __name__ == "__main__":
    res = 64
    sdf_path = '/raid/haoran/Project/PartDiffusion/PartDiffusion/dataset/part_sdf/slider_drawer/12231_0.h5'
    bbox_path = '/raid/haoran/Project/PartDiffusion/PartDiffusion/dataset/part_bbox/slider_drawer/12231_0_bbox.txt'
    h5_f = h5py.File(sdf_path, 'r')
    sdf = h5_f['pc_sdf_sample'][:].astype(np.float32)
    sdf = torch.Tensor(sdf).view(res, res, res)
    bbox = np.loadtxt(bbox_path)

    RT, mesh = calculate_rel_RT(sdf, bbox, ret_mesh=True)
    print(RT)

    # rotate the mesh
    mesh.apply_transform(RT)
    # save the mesh
    mesh.export('/raid/haoran/Project/PartDiffusion/PartDiffusion/dataset/12231_0.obj')


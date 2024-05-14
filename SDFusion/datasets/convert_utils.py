import os
import h5py
import trimesh
import numpy as np
# import marching_cubes as mcubes
import mcubes
import imageio
import einops
from einops import rearrange, repeat
# from skimage import measure
from termcolor import cprint

import torch
import torchvision.utils as vutils

import trimesh
import skimage

# import pytorch3d
# import pytorch3d.io
# from pytorch3d.structures import Pointclouds, Meshes

from kaolin.metrics.trianglemesh import point_to_mesh_distance
from kaolin.ops.mesh import check_sign, index_vertices_by_faces

def try_level_marching_cubes(sdf, level, spacing):
    try:
        vertices, faces, normals, _ = skimage.measure.marching_cubes(sdf, level=level, spacing=spacing)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
        return mesh
    except:
        level += 0.0025
        return try_level_marching_cubes(sdf, level, spacing)

def sdf_to_mesh_trimesh(sdf, level=0.02, spacing=(0.01,0.01,0.01)):
    if torch.is_tensor(sdf):
        sdf = sdf.detach().cpu().numpy()

    # vertices, faces, normals, _ = skimage.measure.marching_cubes(sdf, level=level, spacing=spacing)
    # mesh_mar = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)

    mesh_mar = try_level_marching_cubes(sdf, level, spacing)

    mar_bounding = mesh_mar.bounding_box
    mar_cen = mesh_mar.bounding_box.centroid
    new_vertices = mesh_mar.vertices - mar_cen
    mesh = trimesh.Trimesh(new_vertices, mesh_mar.faces)

    return mesh

def mesh_to_sdf(mesh, res=64, padding=0.2, trunc=0.2, device='cuda'):
    def to_tensor(data, device='cuda'):
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        else:
            raise NotImplementedError()

    def compute_unit_cube_spacing_padding(mesh, padding, voxel_resolution):
        """
        returns spacing for marching cube
        add padding, attention!!! padding must be same with mesh_to_voxels_padding
        """
        spacing = (np.max(mesh.bounding_box.extents) + padding) / voxel_resolution
        return spacing

    def scale_to_unit_cube_padding(mesh: trimesh.Trimesh, padding: float):
        """
        add padding
        """
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump().sum()

        mesh.apply_translation(-mesh.bounding_box.centroid)
        mesh.apply_scale(2 / np.max(mesh.bounding_box.extents)) # first scale to unit cube, i.e., max(extents) = 2
        # print(np.max(np.abs(mesh.bounds)), np.max(mesh.bounding_box.extents), np.max(mesh.extents))
        mesh.apply_scale(2 / (2 + padding)) # then padding 0.2
        # print(np.max(np.abs(mesh.bounds)), np.max(mesh.bounding_box.extents), np.max(mesh.extents))

        return mesh

    class KaolinMeshModel():
        def __init__(self, store_meshes=None, device="cuda"):
            """
            Args:
                `store_meshes` Optional, `list` of `Mesh`.
            """
            self.device = device
            if store_meshes is not None:
                self.update_meshes(store_meshes)
            
        def update_meshes(self, meshes):
            if meshes is not None:
                self.object_mesh_list = []
                self.object_verts_list = []
                self.object_faces_list = []
                self.object_face_verts_list = []
                for mesh in meshes:
                    self.object_mesh_list.append(mesh)
                    self.object_verts_list.append(torch.Tensor(self.object_mesh_list[-1].vertices).to(self.device))
                    self.object_faces_list.append(torch.Tensor(self.object_mesh_list[-1].faces).long().to(self.device))
                    self.object_face_verts_list.append(index_vertices_by_faces(self.object_verts_list[-1].unsqueeze(0), self.object_faces_list[-1]))
            self.num_meshes = len(meshes)

        def mesh_points_sd(self, mesh_idx, points, device):
            """
            Compute the signed distance of a specified point cloud (`points`) to a mesh (specified by `mesh_idx`).

            Args:
                `mesh_idx`: Target mesh index in stored.
                `points`: Either `list`(B) of `ndarrays`(N x 3) or `Tensor` (B x N x 3).

            Returns:
                `signed_distance`: `Tensor`(B x N)
            """
            points = to_tensor(points, device)
            verts = self.object_verts_list[mesh_idx].clone().unsqueeze(0).tile((points.shape[0], 1, 1))
            faces = self.object_faces_list[mesh_idx].clone()
            face_verts = self.object_face_verts_list[mesh_idx]
            
            signs = check_sign(verts, faces, points)
            dis, _, _ = point_to_mesh_distance(points, face_verts)      # Note: The calculated distance is the squared euclidean distance.
            dis = torch.sqrt(dis)                  
            return torch.where(signs, -dis, dis)
    
    voxel_resolution = res

    # save_spacing_centroid_dic = {}
    # ###### calculate spacing before mesh scale to unit cube
    # spacing = compute_unit_cube_spacing_padding(mesh, padding, voxel_resolution)
    # save_spacing_centroid_dic['spacing'] = str(spacing)
    # save_spacing_centroid_dic['padding'] = str(padding)
    # save_spacing_centroid_dic['centroid'] = np.array(mesh.bounding_box.centroid).tolist()

    mesh = scale_to_unit_cube_padding(mesh, padding)

    # voxelize unit cube
    xs = np.linspace(-1, 1, voxel_resolution)
    ys = np.linspace(-1, 1, voxel_resolution)
    zs = np.linspace(-1, 1, voxel_resolution)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing='ij')
    points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float32).cuda()

    obj_meshes = []
    obj_meshes.append(mesh)
    kl = KaolinMeshModel(store_meshes=obj_meshes, device=device)
    sdf = kl.mesh_points_sd(0, points.unsqueeze(0).contiguous(), device)
    sdf = sdf.reshape((1, voxel_resolution, voxel_resolution, voxel_resolution)).detach()

    sdf = sdf.clamp(-trunc, trunc)
    
    return sdf
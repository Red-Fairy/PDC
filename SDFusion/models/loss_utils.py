import torch
import os

import numpy as np
import trimesh
import torch.nn.functional as F
import open3d

from datasets.convert_utils import sdf_to_mesh_trimesh

from torch import nn

from utils.util_3d import init_mesh_renderer, render_sdf

def get_points_from_bbox(part_translation, part_extent, sample_count_per_axis=25, return_indices=False):
    '''
    part_translation: (B, 3)
    part_extent: (B, 3)
    Fill the unit cube with sample_count_per_axis^3 points ([-1, 1]^3)
    for each part, return the idx of the points that are outside the part
    return: list of tensors with length B, each tensor is (N, 3), N may be different for different parts
    '''
    B = part_translation.shape[0]
    points = np.array([[x, y, z] for x in np.linspace(-1, 1, sample_count_per_axis) 
                        for y in np.linspace(-1, 1, sample_count_per_axis)
                        for z in np.linspace(-1, 1, sample_count_per_axis)]) # (sample_count_per_axis^3, 3)
    points = torch.from_numpy(points).float().to(part_translation.device) # (N, 3)

    # Calculate bounding box corners for each part
    min_corners = part_translation - part_extent / 2
    max_corners = part_translation + part_extent / 2

    outsiders = []

    # Check each point for each part
    for i in range(B):
        # For the current part, find points outside its bounding box
        outside_idx = torch.any((points < min_corners[i]) | (points > max_corners[i]), dim=1)
        if return_indices:
            outsiders.append(outside_idx)
        else:
            outsiders.append(points[outside_idx])
    
    return outsiders

def get_collision_loss(sdf, ply, ply_translation, ply_rotation,
                       part_extent, part_translation, 
                       move_limit=None, move_axis=None, move_samples=32, res=64,
                       sdf_scale=None,
                       loss_collision_weight=1.0, margin=0.005, 
                       use_bbox=False, linspace=False):
    '''
    sdf: sdf values, (B, 1, res, res, res), multiple generated sdf with the same point cloud condition
    ply: point cloud, (1, 3, N)
    ply_translation: translation of the ply, (1, 3)
    ply_rotation: rotation of the ply, (1, 4, 4)
    part_extent: extent of the part, (1, 3)
    part_translation: translation of the part, (1, 3)
    move_limit: [min, max]
    sdf_scale: if None, calculate it on-the-fly
    use_bbox: if True, use the bounding box of the part to sample points, add to sampled points to the point cloud
    linspace: if True, use linspace to sample the distance when moving the part
    '''

    B = 1 if move_limit is None else move_samples
    sdf = sdf.repeat(B, 1, 1, 1, 1) # (B, 1, res, res, res)

    if sdf_scale is None: # calculate the scale on-the-fly
        spacing = (2./res, 2./res, 2./res)
        mesh = sdf_to_mesh_trimesh(sdf[0][0], spacing=spacing)
        sdf_scale = np.max(mesh.extents)
        print(sdf_scale)
    
    scale = torch.max(part_extent) / sdf_scale # scalar

    # -2) rotate the point cloud to the canonical pose
    # print(ply.shape, ply_rotation.shape)
    ply = torch.matmul(ply_rotation[:, :3, :3].permute(0, 2, 1), ply) # (1, 3, N)

    # -1) move the point cloud back to original place
    ply = ply + ply_translation.view(1, 3, 1) # (1, 3, N)

    # 0) if use_bbox, add the points from the bounding box of the part to the point cloud
    if use_bbox:
        outsider_points = get_points_from_bbox(part_translation, part_extent)[0].transpose(0, 1).view(1, 3, -1) # (1, 3, N)
        ply = torch.cat([ply, outsider_points], dim=-1) # (1, 3, N')

    # 1) translate the point cloud to the part coordinate
    ply_transformed = (ply - part_translation.view(1, 3, 1)).expand(B, -1, -1) # (B, 3, N)

    # 2) if use mobility constraint, apply the constraint, randomly sample a distance
    if move_limit is not None:
        if not linspace:
            dist = torch.rand([B, 1, 1], device=sdf.device) * (move_limit[1] - move_limit[0]) + move_limit[0] # (B, 1, 1)
        else:
            dist = torch.linspace(move_limit[0], move_limit[1], B, device=sdf.device).view(-1, 1, 1)
        dist_vec = move_axis.view(1, 3, 1) * dist.repeat(1, 3, 1) # (B, 3, 1)
        ply_transformed = ply_transformed - dist_vec # (B, 3, N) move the part, i.e., reversely move the point cloud

    ply_transformed = ply_transformed / scale # (B, 3, N)
    ply_transformed = ply_transformed.transpose(1, 2) # (B, N, 3)

    # export the point cloud for debug, save 
    # path = os.path.join('point_cloud.ply')
    # ply_file = open3d.geometry.PointCloud()
    # ply_file.points = open3d.utility.Vector3dVector(ply_transformed[27].cpu().numpy())
    # open3d.io.write_point_cloud(path, ply_file)
    
    # 3) query the sdf value at the transformed point cloud
    # input: (B, 1, res_sdf, res_sdf, res_sdf), (B, 1, 1, N, 3) -> (B, 1, 1, 1, N)
    sdf_ply = F.grid_sample(sdf, ply_transformed.unsqueeze(1).unsqueeze(1), align_corners=True, padding_mode='border').squeeze(1).squeeze(1).squeeze(1) # (B, N)

    # for the 27th mesh, print its sampled sdf bucket in range [-1, 1] each bucket has 0.005 width
    # sampled_values = sdf_ply[27]
    # rg = torch.arange(-1, 1, 0.005)
    # for i in range(rg.shape[0] - 1):
    #     print(f"Bucket from {rg[i]} to {rg[i+1]}: {torch.sum((sampled_values >= rg[i]) & (sampled_values < rg[i+1]))}")

    # 4) calculate the collision loss
    loss_collision = torch.sum(F.relu(-sdf_ply-margin), dim=1)[0] # (B, N) -> (B, 1), return as a vector

    return loss_collision * loss_collision_weight

def get_surface_points_from_bbox(part_translation, part_extent, sample_count_per_axis=50):
    '''
    part_translation: (B, 3)
    part_extent: (B, 3)
    for each part, return sample_count_per_axis^2 points on each face of the part
    '''
    B = part_translation.shape[0]
    points_z = np.array([[[x, y, z] for x in np.linspace(-1, 1, sample_count_per_axis) 
                        for y in np.linspace(-1, 1, sample_count_per_axis) for z in [-1, 1]]]) # (1, sample_count_per_axis^2, 3)
    points_x = np.array([[[x, y, z] for x in [-1, 1] for y in np.linspace(-1, 1, sample_count_per_axis) 
                        for z in np.linspace(-1, 1, sample_count_per_axis)]]) # (1, 2*sample_count_per_axis^2, 3)
    points_y = np.array([[[x, y, z] for x in np.linspace(-1, 1, sample_count_per_axis) for y in [-1, 1]
                        for z in np.linspace(-1, 1, sample_count_per_axis)]]) # (1, 2*sample_count_per_axis^2, 3)
    points = np.concatenate([points_z, points_x, points_y]) # (1, 6*sample_count_per_axis^2, 3)
    points = points.repeat(B, 1, 1) # (B, 6*sample_count_per_axis^2, 3)
    points = points * part_extent.view(B, 1, 3) / 2 + part_translation.view(B, 1, 3) # (B, 6*sample_count_per_axis^2, 3)

    return torch.from_numpy(points).float()

# def get_collision_loss_bbox(sdf, part_extent, part_translation, 
#                             move_limit=None, move_axis=None, sdf_scale=4/2.2, 
#                             loss_collision_weight=1.0, margin=0.001,
#                             sample_count_per_axis=25):
#     '''
#     sdf: sdf values, (B, 1, res, res, res)
#     part_extent: extent of the part, (B, 3)
#     part_translation: translation of the part, (B, 3)
#     move_limit: (B, 2)
#     '''

#     B = sdf.shape[0]
#     scale = torch.max(part_extent) / sdf_scale # scalar

#     # 2) if use mobility constraint, apply the constraint, randomly sample a distance
#     if move_limit is not None:
#         dist = (torch.rand([B], device=sdf.device) * (move_limit[:, 1] - move_limit[:, 0]) + move_limit[:, 0]).view(-1, 1, 1) # (B, 1, 1)
#         dist_vec = move_axis.view(1, 3, 1) * dist.repeat(1, 3, 1) # (B, 3, 1)
#         part_translation = part_translation.unsqueeze(-1).repeat(B, 1, 1) + dist_vec # (B, 3, 1)

#     # outside_points_indices = get_points_from_bbox(part_translation, part_extent, sample_count_per_axis) # (B, N), N=sample_count_per_axis^3

#     # 3) query the sdf value at the points
#     points = np.array([[[x, y, z] for x in np.linspace(-1, 1, sample_count_per_axis) 
#                         for y in np.linspace(-1, 1, sample_count_per_axis)
#                         for z in np.linspace(-1, 1, sample_count_per_axis)]]) # (1, sample_count_per_axis^3, 3)
#     min_corners = part_translation - part_extent / 2
#     max_corners = part_translation + part_extent / 2

#     outsider_indices = []

#     for i in range(B):
#         # For the current part, find points outside its bounding box
#         outside_idx = np.where(
#             np.any(
#                 (points[0] < min_corners[i]) | (points[0] > max_corners[i]),
#                 axis=1
#             )
#         )[0]
#         outsider_indices.append(outside_idx)
    
#     outsider_indices = torch.from_numpy(np.array(outsider_indices)).long().to(sdf.device)
        
#     # 4) query the sdf values


#     # 4) calculate the collision loss
#     loss_collision_part = torch.sum(F.relu(-sdf[:, :, outside_points_indices] - margin), dim=1) # (B, N) -> (B, 1), return as a vector

#     return loss_collision_part * loss_collision_weight

class VQLoss(nn.Module):
    def __init__(self, codebook_weight=1.0):
        super().__init__()
        self.codebook_weight = codebook_weight

    def forward(self, codebook_loss, inputs, reconstructions, split="train"):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())

        nll_loss = rec_loss
        nll_loss = torch.mean(nll_loss)

        loss = nll_loss + self.codebook_weight * codebook_loss.mean()

        log = {
            "total": loss.clone().detach().mean(),
            "codebook": codebook_loss.detach().mean(),
            "nll": nll_loss.detach().mean(),
            "rec": rec_loss.detach().mean(),
        }

        return loss, log
import torch
import os

import numpy as np
import trimesh
import torch.nn.functional as F
import open3d

from datasets.convert_utils import sdf_to_mesh_trimesh, mesh_to_sdf

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

def get_physical_loss(sdf, ply, ply_translation, ply_rotation,
                       part_extent, part_translation, 
                       move_limit=None, move_axis=None, move_origin=None, 
                       move_type=None, move_samples=32, 
                       res=64,
                       scale_mode='max_extent',
                       margin=1/256,
                       use_bbox=False, linspace=False,
                       loss_collision_weight=1.0,
                       loss_contact_weight=10000.0):
    '''
    sdf: sdf values, (B, 1, res, res, res), multiple generated sdf with the same point cloud condition
    ply: point cloud, (1, 3, N)
    ply_translation: translation of the ply, (1, 3)
    ply_rotation: rotation of the ply, (1, 4, 4)
    part_extent: extent of the part, (1, 3)
    part_translation: translation of the part, (1, 3)
    move_limit: [min, max]
    scale_mode: volume or max_extent
    use_bbox: if True, use the bounding box of the part to sample points, add to sampled points to the point cloud
    linspace: if True, use linspace to sample the distance when moving the part
    '''

    B = 1 if move_limit is None else move_samples
    sdf = sdf.repeat(B, 1, 1, 1, 1) # (B, 1, res, res, res)

    spacing = (2./res, 2./res, 2./res)
    mesh = sdf_to_mesh_trimesh(sdf[0][0], spacing=spacing)
    if scale_mode == 'volume':
        scale = ((torch.prod(part_extent.prod(dim=-1)) / np.prod(mesh.extents)) ** (1/3)).item()
    else:
        scale = (torch.max(part_extent) / torch.max(torch.tensor(mesh.extents))).item()

    # -2) rotate the point cloud to the canonical pose
    # print(ply.shape, ply_rotation.shape)
    ply = torch.matmul(ply_rotation[:, :3, :3].permute(0, 2, 1), ply) # (1, 3, N)

    # -1) move the point cloud back to original place
    ply = ply + ply_translation.view(1, 3, 1) # (1, 3, N)

    # 0) if use_bbox, add the points from the bounding box of the part to the point cloud
    if use_bbox:
        outsider_points = get_points_from_bbox(part_translation, part_extent)[0].transpose(0, 1).view(1, 3, -1) # (1, 3, N)
        ply = torch.cat([ply, outsider_points], dim=-1) # (1, 3, N')

    # 1) if use mobility constraint, apply the constraint, randomly sample a distance
    if move_limit is not None:
        if move_type == 'translation':
            if not linspace:
                dist = torch.rand([B, 1, 1], device=sdf.device) * (move_limit[1] - move_limit[0]) + move_limit[0] # (B, 1, 1)
            else:
                dist = torch.linspace(move_limit[0], move_limit[1], B, device=sdf.device).view(-1, 1, 1)
            dist_vec = move_axis.view(1, 3, 1) * dist.repeat(1, 3, 1) # (B, 3, 1)
            ply = ply.expand(B, -1, -1) - dist_vec # (B, 3, N) move the part, i.e., reversely move the point cloud
        elif move_type == 'rotation':
            if not linspace:
                angle = torch.rand([B, 1, 1], device=sdf.device) * (move_limit[1] - move_limit[0]) + move_limit[0]
            else:
                angle = torch.linspace(move_limit[0], move_limit[1], B, device=sdf.device).view(-1, 1, 1)
            angle = angle / 180 * np.pi # convert to radians
            ply = ply - move_origin.view(1, 3, 1) # translate ply to the origin
            K = torch.tensor([[0, -move_axis[2], move_axis[1]],
                              [move_axis[2], 0, -move_axis[0]],
                              [-move_axis[1], move_axis[0], 0]], device=sdf.device) # construct the rotation matrix
            R = torch.eye(3, device=sdf.device) + torch.sin(-angle) * K + (1 - torch.cos(-angle)) * torch.matmul(K, K) # (B, 3, 3)
            ply = torch.matmul(R, ply) # (B, 3, N)
            ply = ply + move_origin.view(1, 3, 1) # translate ply back
            # debug
            # ply_file = open3d.geometry.PointCloud()
            # ply_file.points = open3d.utility.Vector3dVector(ply[1].cpu().numpy().T)
            # open3d.io.write_point_cloud('debug.ply', ply_file)
        else:
            assert False

    # 2) translate the point cloud to the part coordinate
    ply = ply - part_translation.view(1, 3, 1) # (B, 3, N)

    ply = ply / scale # (B, 3, N)

    # due to different 3D conventions, ply need to be rotated by 90 degrees along the y-axis
    ply_rotated = torch.bmm(torch.tensor([[0, 0, 1.], [0, 1., 0], [-1., 0, 0]], device=sdf.device, dtype=torch.float32).repeat(B, 1, 1), ply) # (B, 3, N)

    ply_rotated = ply_rotated.transpose(1, 2) # (B, N, 3)
    
    # 3) query the sdf value at the transformed point cloud
    # input: (B, 1, res_sdf, res_sdf, res_sdf), (B, 1, 1, N, 3) -> (B, 1, 1, 1, N)
    sdf_ply = F.grid_sample(sdf, ply_rotated.unsqueeze(1).unsqueeze(1), align_corners=True, padding_mode='border').squeeze(1).squeeze(1).squeeze(1) # (B, N)

    # 4) calculate the collision loss
    loss_collision = torch.sum(torch.max(F.relu(-sdf_ply-margin), dim=0)[0])

    # 4+) calculate the contact loss, punish if all points are outside the part
    loss_contact = torch.sum(torch.min(F.relu(sdf_ply-margin), dim=1)[0])

    # debug, filter the points with positive sampled sdf values
    # ply_file = open3d.geometry.PointCloud()
    # ply_file.points = open3d.utility.Vector3dVector((ply[0].T).cpu().numpy())
    # open3d.io.write_point_cloud('debug_ori_0.ply', ply_file)
    # ply_file.points = open3d.utility.Vector3dVector((ply[0].T).cpu().numpy())
    # open3d.io.write_point_cloud('debug_ori_moved.ply', ply_file)
    # mesh = sdf_to_mesh_trimesh(sdf[0][0], spacing=(2./res, 2./res, 2./res))
    # sdf = mesh_to_sdf(mesh).unsqueeze(0).repeat(B,1,1,1,1)
    # mesh.apply_translation(-mesh.bounding_box.centroid)
    # mesh.export('debug.obj')
    # ply_file = open3d.geometry.PointCloud()
    # sdf_max = torch.max(F.relu(-sdf_ply-margin), dim=0)[0]
    # sdf_max = sdf_ply-margin
    # ply_file.points = open3d.utility.Vector3dVector((ply[0].T)[torch.where(sdf_max[0] < 0)].cpu().numpy())
    # open3d.io.write_point_cloud('debug_ori_collision.ply', ply_file)
    # ply_file.points = open3d.utility.Vector3dVector((ply[-1].T)[torch.where(sdf_max[-1] < 0)].cpu().numpy())
    # open3d.io.write_point_cloud('debug_moved_collision.ply', ply_file)

    return loss_collision * loss_collision_weight, loss_contact * loss_contact_weight

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
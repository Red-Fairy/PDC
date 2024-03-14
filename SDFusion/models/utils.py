import torch
import os
from collections import OrderedDict
from functools import partial

from utils.util_3d import sdf_to_mesh_trimesh
import numpy as np
import trimesh
import torch.nn.functional as F

from utils.util_3d import init_mesh_renderer, render_sdf

def get_collision_loss(sdf, ply, ply_translation, part_extent, part_translation, 
                       move_limit=None, move_axis=None, sdf_scale=4/2.2, 
                       loss_collision_weight=1.0, margin=0.001):
    '''
    sdf: sdf values, (B, 1, res, res, res), multiple generated sdf with the same point cloud condition
    ply: point cloud, (1, 3, N)
    ply_translation: translation of the ply, (1, 3)
    part_extent: extent of the part, (1, 3)
    part_translation: translation of the part, (1, 3)
    move_limit: [min, max]
    '''

    B = sdf.shape[0]
    scale = torch.max(part_extent) / sdf_scale # scalar

    # 1) translate the point cloud
    ply_transformed = ((ply + ply_translation.view(1, 3, 1) - part_translation.view(1, 3, 1))).expand(B, -1, -1) # (B, 3, N)

    # 2) if use mobility constraint, apply the constraint, randomly sample a distance
    if move_limit is not None:
        dist = torch.rand([B, 1, 1], device=sdf.device) * (move_limit[1] - move_limit[0]) + move_limit[0] # (B, 1, 1)
        dist_vec = move_axis.view(1, 3, 1) * dist.repeat(1, 3, 1) # (B, 3, 1)
        ply_transformed = ply_transformed - dist_vec # (B, 3, N) move the part, i.e., reversely move the point cloud

    ply_transformed = ply_transformed / scale # (B, 3, N)
    ply_transformed = ply_transformed.transpose(1, 2) # (B, N, 3)
    
    # 3) query the sdf value at the transformed point cloud
    # input: (1, 1, res_sdf, res_sdf, res_sdf), (B, 1, 1, N, 3) -> (B, 1, 1, 1, N)
    sdf_ply = F.grid_sample(sdf, ply_transformed.unsqueeze(1).unsqueeze(1), align_corners=True).squeeze(1).squeeze(1).squeeze(1) # (B, N)

    # 4) calculate the collision loss
    loss_collision = torch.mean(torch.sum(F.relu(-sdf_ply-margin), dim=1)[0]) # (B, N) -> (1, N) -> scalar

    return loss_collision * loss_collision_weight


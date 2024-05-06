import os
import open3d
from datasets.convert_utils import sdf_to_mesh_trimesh, mesh_to_sdf
import torch
import numpy as np
import torch.nn.functional as F
import trimesh
import argparse

def get_physical_loss(sdf, ply, scale, part_translation, 
                       move_limit=None, move_axis=None, move_origin=None, 
                       move_type=None, move_samples=32,
                       margin=1/256):
    '''
    sdf: sdf values, (B, 1, res, res, res), multiple generated sdf with the same point cloud condition
    ply: point cloud, (1, 3, N)
    part_translation: translation of the part, (1, 3)
    move_limit: [min, max]
    scale_mode: volume or max_extent
    use_bbox: if True, use the bounding box of the part to sample points, add to sampled points to the point cloud
    linspace: if True, use linspace to sample the distance when moving the part
    '''

    B = 1 if move_limit is None else move_samples
    sdf = sdf.repeat(B, 1, 1, 1, 1) # (B, 1, res, res, res)

    # 1) if use mobility constraint, apply the constraint, randomly sample a distance
    if move_limit is not None:
        if move_type == 'translation':
            dist = torch.linspace(move_limit[0], move_limit[1], B, device=sdf.device).view(-1, 1, 1)
            dist_vec = move_axis.view(1, 3, 1) * dist.repeat(1, 3, 1) # (B, 3, 1)
            ply = ply.expand(B, -1, -1) - dist_vec # (B, 3, N) move the part, i.e., reversely move the point cloud
        elif move_type == 'rotation':
            angle = torch.linspace(move_limit[0], move_limit[1], B, device=sdf.device).view(-1, 1, 1)
            angle = angle / 180 * np.pi # convert to radians
            ply = ply - move_origin.view(1, 3, 1) # translate ply to the origin
            K = torch.tensor([[0, -move_axis[2], move_axis[1]],
                              [move_axis[2], 0, -move_axis[0]],
                              [-move_axis[1], move_axis[0], 0]], device=sdf.device) # construct the rotation matrix
            R = torch.eye(3, device=sdf.device) + torch.sin(-angle) * K + (1 - torch.cos(-angle)) * torch.matmul(K, K) # (B, 3, 3)
            ply = torch.matmul(R, ply) # (B, 3, N)
            ply = ply + move_origin.view(1, 3, 1) # translate ply back
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
    sdf_ply = F.grid_sample(sdf, ply_rotated.unsqueeze(1).unsqueeze(1), align_corners=True, padding_mode='zeros').squeeze(1).squeeze(1).squeeze(1) # (B, N)

    # 4) calculate the collision loss
    loss_collision = torch.sum(torch.max(F.relu(-sdf_ply-margin), dim=0)[0])
    loss_contact = torch.sum(torch.min(F.relu(sdf_ply), dim=1)[0])

    return loss_collision < 1e-4 and loss_contact < 1e-4

def physical_feasibility(mesh: trimesh.Trimesh, pcd: np.array,
                         res: int=256, padding: float=0.2, margin: float=1/256,
                         grid_length=1/256, steps=11):
    '''
    mesh: trimesh mesh
    pcd: point cloud, (N, 3)
    '''
    # 1) get the sdf value of the point cloud
    mesh_max_extent = np.max(mesh.bounding_box.extents)
    translation = torch.tensor(mesh.bounding_box.centroid).unsqueeze(0).to('cuda') # (1, 3)
    scale = (2. / mesh_max_extent) * (2 / (2 + padding))
    spacing = (2./res, 2./res, 2./res)
    sdf = mesh_to_sdf(mesh, res=res, spacing=spacing, padding=padding)
    sdf = torch.tensor(sdf).unsqueeze(0).unsqueeze(0).to('cuda') # (1, 1, res, res, res)
    pcd = torch.tensor(pcd).unsqueeze(0).permute(0, 2, 1).to('cuda') # (1, 3, N)

    # grid search the physical feasibility around the translation
    # build a grid
    x = torch.linspace(-grid_length, grid_length, steps).to('cuda')
    y = torch.linspace(-grid_length, grid_length, steps).to('cuda')
    z = torch.linspace(-grid_length, grid_length, steps).to('cuda')
    X, Y, Z = torch.meshgrid(x, y, z)
    X, Y, Z = X.contiguous().view(-1), Y.contiguous().view(-1), Z.contiguous().view(-1)
    # sort with the absolute distance to the origin
    zipped = sorted(zip(X, Y, Z), key=lambda x: torch.norm(torch.tensor(x)))
    for (dx, dy, dz) in zipped:
        part_translation = translation + torch.tensor([dx, dy, dz]).unsqueeze(0).to('cuda')
        if get_physical_loss(sdf, pcd,scale, part_translation, 
                        move_limit=None, move_axis=None, move_origin=None, 
                        move_type=None, move_samples=32,
                        margin=margin):
            print(f'Physical feasible at translation: {part_translation}, translation: {part_translation[0].cpu().numpy()}')
            return True
        
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_root', type=str, default='./logs/slider_drawer-ply2shape-plyrot-scale3-lr0.00001/test_250000_rotate0.0_scale3.0_eta0.0_steps50_volume_mobility_diversity_margin256-haoran/meshes_canonical_selected')
    parser.add_argument('--gt_root', type=str, default='../../data-rundong/PartDiffusion/dataset/')
    args = parser.parse_args()

    test_obj_files = os.listdir(args.test_root)
    test_pcd_files = [f.replace('.obj', '.ply') for f in test_obj_files]

    test_obj_files = [os.path.join(args.test_root, f) for f in test_obj_files]
    test_pcd_files = [os.path.join(args.gt_root, 'part_ply_fps', 'slider_drawer', f) for f in test_pcd_files]

    for (obj_file, pcd_file) in zip(test_obj_files, test_pcd_files):
        obj = trimesh.load(obj_file)
        pcd = open3d.io.read_point_cloud(pcd_file)
        pcd = np.asarray(pcd.points) # (N, 3)

        print(f'Physical feasibility of {obj_file}: {physical_feasibility(obj, pcd)}')
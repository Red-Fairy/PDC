import os
import open3d
import sys
sys.path.append('.')
from utils import sdf_to_mesh_trimesh, mesh_to_sdf
import torch
import numpy as np
import torch.nn.functional as F
import trimesh
from tqdm import tqdm
import argparse
import skimage

global_cnt = 0

def visualize_sdf(sdf: torch.Tensor, thres=0.1):
    # create a point cloud that fills the unit cube, and query the sdf value
    # keep the point with value < threshold
    res = sdf.shape[-1]
    x = torch.linspace(-1, 1, res)
    y = torch.linspace(-1, 1, res)
    z = torch.linspace(-1, 1, res)
    X, Y, Z = torch.meshgrid(x, y, z)
    X, Y, Z = X.contiguous().view(-1), Y.contiguous().view(-1), Z.contiguous().view(-1)
    points = torch.stack([X, Y, Z], dim=1).to(sdf.device)
    sdf_values = F.grid_sample(sdf, points.unsqueeze(0).unsqueeze(0).unsqueeze(0), align_corners=True, padding_mode='border').squeeze(0).squeeze(0).squeeze(0).squeeze(0)
    print(sdf.min(), sdf.max())
    points = points[sdf_values < thres]
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points.cpu().numpy())
    open3d.io.write_point_cloud('debug.ply', pcd)

def get_physical_loss(sdf, ply, scale, part_translation, 
                       move_limit=None, move_axis=None, move_origin=None, 
                       move_type=None, move_samples=32,
                       collision_margin=1/256, contact_margin=1/256):
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

    ply = ply * scale # (B, 3, N)

    # due to different 3D conventions, ply need to be rotated by 90 degrees along the y-axis
    ply_rotated = torch.bmm(torch.tensor([[0, 0, 1.], [0, 1., 0], [-1., 0, 0]], device=sdf.device, dtype=torch.float32).repeat(B, 1, 1), ply.float().to(sdf.device)) # (B, 3, N)

    ply_rotated = ply_rotated.transpose(1, 2) # (B, N, 3)
    
    # 3) query the sdf value at the transformed point cloud
    # input: (B, 1, res_sdf, res_sdf, res_sdf), (B, 1, 1, N, 3) -> (B, 1, 1, 1, N)
    sdf_ply = F.grid_sample(sdf, ply_rotated.unsqueeze(1).unsqueeze(1), align_corners=True, padding_mode='border').squeeze(1).squeeze(1).squeeze(1) # (B, N)

    # debug, filter the points with positive sampled sdf values
    # ply_file = open3d.geometry.PointCloud()
    # res = 128
    # ply_file.points = open3d.utility.Vector3dVector((ply[0].T).cpu().numpy())
    # open3d.io.write_point_cloud('debug_ori_0.ply', ply_file)
    # ply_file.points = open3d.utility.Vector3dVector((ply[-1].T).cpu().numpy())
    # open3d.io.write_point_cloud('debug_ori_moved.ply', ply_file)
    # mesh = sdf_to_mesh_trimesh(sdf[0][0], spacing=(2./res, 2./res, 2./res))
    # sdf = mesh_to_sdf(mesh).unsqueeze(0).repeat(B,1,1,1,1)
    # mesh.apply_translation(-mesh.bounding_box.centroid)
    # mesh.export('debug.obj')
    # sdf_max = F.relu(-sdf_ply-collision_margin)
    # ply_file.points = open3d.utility.Vector3dVector((ply[0].T)[torch.where(sdf_max[0] > 0)].cpu().numpy())
    # open3d.io.write_point_cloud('debug_ori_collision.ply', ply_file)
    # ply_file.points = open3d.utility.Vector3dVector((ply[-1].T)[torch.where(sdf_max[-1] > 0)].cpu().numpy())
    # open3d.io.write_point_cloud('debug_moved_collision.ply', ply_file)
    # exit()

    # 4) calculate the collision loss
    loss_collision = torch.sum(F.relu(-sdf_ply-collision_margin)) / B

    loss_contact = torch.sum(torch.min(F.relu(sdf_ply-contact_margin), dim=1)[0]) / B

    return loss_collision.item(), loss_contact.item()

def passed_physical_feasibility(contact_loss, collision_loss, 
                                contact_thres=1e-5, collision_thres=1e-5):
    return contact_loss < contact_thres and collision_loss < collision_thres

def physical_feasibility(mesh: trimesh.Trimesh, pcd: torch.Tensor,
                         res: int=128, padding: float=0.2, 
                         collision_margin=1/128, contact_margin=1/128,
                         grid_length=1/128, steps=(6, 6, 6), step_thres=1,
                         contact_thres=1e-5, collision_thres=1e-5,
                         move_type='translation', move_limit=None, move_axis=None, move_origin=None,
                         device='cuda:0'):
    '''
    mesh: trimesh mesh
    pcd: point cloud, (N, 3)
    '''
    # 1) get the sdf value of the point cloud
    mesh_max_extent = np.max(mesh.bounding_box.extents)
    translation = torch.tensor(mesh.bounding_box.centroid).unsqueeze(0).to(device) # (1, 3)
    scale = torch.tensor((2. / mesh_max_extent) * (2 / (2 + padding)), dtype=torch.float32).to(device)  # scale the part to the unit cube
    sdf = mesh_to_sdf(mesh, res=res, padding=padding, device=device).unsqueeze(0) # (1, 1, res, res, res)

    pcd = pcd.unsqueeze(0).permute(0, 2, 1).to(device)  # (1, 3, N)
    # grid search the physical feasibility around the translation
    x = torch.linspace(-grid_length, grid_length, steps[0]).to(device) 
    y = torch.linspace(-grid_length, grid_length, steps[1]).to(device) 
    z = torch.linspace(-grid_length, grid_length, steps[2]).to(device) 
    X, Y, Z = torch.meshgrid(x, y, z)
    X, Y, Z = X.contiguous().view(-1), Y.contiguous().view(-1), Z.contiguous().view(-1)
    # sort with the absolute distance to the origin
    zipped = sorted(zip(X, Y, Z), key=lambda x: torch.norm(torch.tensor(x)))
    for (dx, dy, dz) in zipped:
        part_translation = translation + torch.tensor([dx, dy, dz]).unsqueeze(0).to(device)
        contact_loss, collsion_loss = get_physical_loss(sdf, pcd, scale, part_translation, 
                                        move_limit=move_limit, move_axis=move_axis, move_origin=move_origin,
                                        move_type=move_type, move_samples=32,
                                        collision_margin=collision_margin, contact_margin=contact_margin)
        print(f'{contact_loss:.4f}, {collsion_loss:.4f}')
        if passed_physical_feasibility(contact_loss, collsion_loss, contact_thres, collision_thres):
                # make sure that at the position, the part cannot move further on x and y axis, (but allow move one step)
                for ddx, ddy in [(0, i*grid_length) for i in range(1, step_thres+1)] + [(0, -i*grid_length) for i in range(1, step_thres+1)]:
                    part_translation_dd = translation + torch.tensor([dx+ddx, dy+ddy, dz]).unsqueeze(0).to(device)
                    contact_loss, collsion_loss = get_physical_loss(sdf, pcd, scale, part_translation_dd, 
                                                move_limit=move_limit, move_axis=move_axis, move_origin=move_origin,
                                                move_type=move_type, move_samples=32,
                                                collision_margin=collision_margin, contact_margin=contact_margin)
                    if not passed_physical_feasibility(contact_loss, collsion_loss, contact_thres, collision_thres):
                        print(f'Physical feasible at translation: {part_translation[0].cpu().numpy()}')
                        return 1
                print(f'Physical feasible at translation: {part_translation[0].cpu().numpy()}' + \
                    f'but also for all vertical movement. Not physical feasible.')
                return 0
    print(f'Not physical feasible for all translations.')
    return -1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_root', type=str, default='../../part_meshes_recon/slider_drawer/')
    parser.add_argument('--gt_root', type=str, default='/mnt/data-rundong/PartDiffusion/dataset/')
    parser.add_argument('--test_list', type=str, default='/mnt/data-rundong/PartDiffusion/data_lists/test/slider_drawer.txt')
    parser.add_argument('--gpu_id', type=int, default=7)
    args = parser.parse_args()

    device = f'cuda:{args.gpu_id}'

    good_objs = []
    all_not_feasible_objs = []
    multiple_feasible_objs = []

    test_obj_files = [x for x in os.listdir(args.test_root) if x.endswith('.obj')]
    with open(args.test_list, 'r') as f:
        test_filenames = f.readlines()
    test_filenames = [f.strip() for f in test_filenames]
    test_obj_files = [f for f in test_obj_files if f.split('.')[0] in test_filenames]
    test_pcd_files = [f.replace('.obj', '.ply') for f in test_obj_files]

    test_obj_files = [os.path.join(args.test_root, f) for f in test_obj_files]
    test_pcd_files = [os.path.join(args.gt_root, 'part_ply_fps', 'slider_drawer', f) for f in test_pcd_files]

    for (obj_file, pcd_file) in zip(test_obj_files, test_pcd_files):
        # if not any([x in obj_file for x in ['29133_0.obj', '32324_1.obj', '34178_1.obj']]):
        #     continue
        obj = trimesh.load(obj_file)
        pcd = open3d.io.read_point_cloud(pcd_file)
        pcd = torch.tensor(np.array(pcd.points), dtype=torch.float32).to(device) # (N, 3)
        
        print(f'Physical feasibility of {os.path.basename(obj_file)}')
        result = physical_feasibility(obj, pcd, device=device,
                                      grid_length=0.01, step_thres=3, 
                                      res=128,
                                      move_type='translation', move_limit=(0, 0.2), 
                                      move_axis=torch.tensor([0., 0., 1.], device=device), 
                                      move_origin=torch.tensor([0., 0., 0.], device=device),
                                      collision_margin=0.01, contact_margin=0.01,
                                      contact_thres=0.15, collision_thres=0.05)

        if result == 1:
            good_objs.append(os.path.basename(obj_file))
        elif result == 0:
            multiple_feasible_objs.append(os.path.basename(obj_file))
        else:
            all_not_feasible_objs.append(os.path.basename(obj_file))

    print(f'Good objects: {good_objs}')
    print(f'Multiple feasible objects: {multiple_feasible_objs}')
    print(f'All not feasible objects: {all_not_feasible_objs}')

if __name__ == '__main__':
    main()

    # export the ply and sdf for debugging
    # global global_cnt
    # ply_file = open3d.geometry.PointCloud()
    # ply_file.points = open3d.utility.Vector3dVector((ply[0].T).cpu().numpy())
    # open3d.io.write_point_cloud('debug_ori_0.ply', ply_file)
    # ply_file.points = open3d.utility.Vector3dVector((ply[-1].T).cpu().numpy())
    # open3d.io.write_point_cloud('debug_ori_moved.ply', ply_file)
    # res=128
    # mesh = sdf_to_mesh_trimesh(sdf[0][0], spacing=(2./res, 2./res, 2./res))
    # sdf = mesh_to_sdf(mesh).unsqueeze(0).repeat(B,1,1,1,1)
    # mesh.apply_translation(-mesh.bounding_box.centroid)
    # mesh.export('debug.obj')
    # # sdf_max = torch.max(F.relu(sdf_ply-margin), dim=0)[0]
    # sdf_max = sdf_ply-margin
    # ply_file.points = open3d.utility.Vector3dVector((ply[0].T)[torch.where(sdf_max[0] < 0)].cpu().numpy())
    # open3d.io.write_point_cloud('debug_ori_collision.ply', ply_file)
    # ply_file.points = open3d.utility.Vector3dVector((ply[-1].T)[torch.where(sdf_max[-1] < 0)].cpu().numpy())
    # open3d.io.write_point_cloud('debug_moved_collision.ply', ply_file)
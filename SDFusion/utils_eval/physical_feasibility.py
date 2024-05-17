import os
import open3d
import sys
sys.path.append('.')
from utils import sdf_to_mesh_trimesh, mesh_to_sdf, Logger
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
        move_origin = torch.tensor(move_origin, device=sdf.device, dtype=torch.float32)
        move_axis = torch.tensor(move_axis, device=sdf.device, dtype=torch.float32)
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
            R = torch.eye(3, device=sdf.device) + torch.sin(angle) * K + (1 - torch.cos(angle)) * torch.matmul(K, K) # (B, 3, 3)
            ply = torch.matmul(R, ply) # (B, 3, N)
            ply = ply + move_origin.view(1, 3, 1) # translate ply back
        else:
            assert False

    # 2) translate the point cloud to the part coordinate
    ply = ply - part_translation.view(1, 3, 1) # (B, 3, N)

    ply = ply / scale # (B, 3, N)

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

def physical_feasibility(mesh: trimesh.Trimesh, pcd: torch.Tensor, logger: Logger,
                         res: int=128, padding: float=0.2,
                         collision_margin=1/128, contact_margin=1/128,
                         grid_length=1/128, steps=(3, 3, 7), step_thres=1,
                         contact_thres=1e-5, collision_thres=1e-5,
                         cat='slider_drawer',
                         move_type='translation', move_limit=None, move_axis=None, 
                         device='cuda:0'):
    '''
    mesh: trimesh mesh
    pcd: point cloud, (N, 3)
    '''
    # 1) get the sdf value of the point cloud
    mesh_extent = mesh.bounding_box.extents
    move_origin = mesh.bounding_box.centroid + mesh.bounding_box.extents / 2
    translation = torch.tensor(mesh.bounding_box.centroid).unsqueeze(0).to(device) # (1, 3)
    sdf = mesh_to_sdf(mesh, res=res, padding=padding, device=device).unsqueeze(0) # (1, 1, res, res, res)

    recon_mesh = sdf_to_mesh_trimesh(sdf[0][0], spacing=(2./res, 2./res, 2./res), level=0.025)
    # move_origin = recon_mesh.bounding_box.centroid + recon_mesh.bounding_box.extents / 2
    if cat == 'hinge_door':
        scale = np.max(mesh_extent) / np.max(recon_mesh.bounding_box.extents)
    else: # scale by volume
        scale = (np.prod(mesh_extent) / np.prod(recon_mesh.bounding_box.extents)) ** (1/3)

    pcd = pcd.unsqueeze(0).permute(0, 2, 1).to(device)  # (1, 3, N)
    # grid search the physical feasibility around the translation
    x = torch.linspace(-grid_length, grid_length, steps[0]).to(device) * (steps[0] - 1) / 2
    y = torch.linspace(-grid_length, grid_length, steps[1]).to(device) * (steps[1] - 1) / 2
    z = torch.linspace(-grid_length, grid_length, steps[2]).to(device) * (steps[2] - 1) / 2
    X, Y, Z = torch.meshgrid(x, y, z)
    X, Y, Z = X.contiguous().view(-1), Y.contiguous().view(-1), Z.contiguous().view(-1)
    # sort with the absolute distance to the origin
    if cat == 'slider_drawer': # sort by x^2 + y^2, if equal, sort by -z
        zipped = sorted(zip(X, Y, Z), key=lambda x: (x[0]**2 + x[2]**2) * 1e8 + abs(x[1]))
    elif cat == 'hinge_door': # sort by x^2 + y^2, if equal, sort by z
        zipped = sorted(zip(X, Y, Z), key=lambda x: (x[0]**2 + x[1]**2) * 1e8 + x[2])
    else: #  sort by x^2 + y^2 + z^2
        zipped = sorted(zip(X, Y, Z), key=lambda x: x[0]**2 + x[1]**2 + x[2]**2)
    # best_loss = 1e5
    # contact_loss_, collsion_loss_ = 0, 0
    for (dx, dy, dz) in zipped:
        part_translation = translation + torch.tensor([dx, dy, dz]).unsqueeze(0).to(device)
        contact_loss, collsion_loss = get_physical_loss(sdf, pcd, scale, part_translation, 
                                        move_limit=move_limit, move_axis=move_axis, move_origin=move_origin,
                                        move_type=move_type, move_samples=32,
                                        collision_margin=collision_margin, contact_margin=contact_margin)
        print(f'{contact_loss:.4f}, {collsion_loss:.4f}')
    #     if contact_loss + collsion_loss < best_loss:
    #         dx_, dy_, dz_ = dx, dy, dz
    #         best_loss = contact_loss + collsion_loss
    #         contact_loss_, collsion_loss_ = contact_loss, collsion_loss
    # if passed_physical_feasibility(contact_loss_, collsion_loss_, contact_thres, collision_thres):
    #     # make sure that at the position, the part cannot move further on x and y axis, (but allow move one step)
    #     if cat == 'slider_drawer':
    #         other_positions_to_check = [(0, i*grid_length, 0) for i in range(1, step_thres+1)] + [(0, -i*grid_length, 0) for i in range(1, step_thres+1)]
    #     elif cat == 'hinge_door':
    #         other_positions_to_check = [(0, 0, -i*grid_length) for i in range(1, step_thres+1)]
    #     else:
    #         other_positions_to_check = [(0, 0, i*grid_length) for i in range(1, step_thres+1)] + [(0, 0, -i*grid_length) for i in range(1, step_thres+1)]
    #     for ddx, ddy, ddz in other_positions_to_check:
    #         part_translation_dd = translation + torch.tensor([dx_+ddx, dy_+ddy, dz_+ddz]).unsqueeze(0).to(device)
    #         contact_loss, collsion_loss = get_physical_loss(sdf, pcd, scale, part_translation_dd, 
    #                                     move_limit=move_limit, move_axis=move_axis, move_origin=move_origin,
    #                                     move_type=move_type, move_samples=32,
    #                                     collision_margin=collision_margin, contact_margin=contact_margin)
    #         if not passed_physical_feasibility(contact_loss, collsion_loss, contact_thres, collision_thres):
    #             logger.log(f'Physical feasible at translation: {part_translation[0].cpu().numpy()}')
    #             return 1
    #     logger.log(f'Physical feasible at translation: {part_translation[0].cpu().numpy()}' + \
    #         f'but also for all movement. Not physical feasible.')
    #     return 0
    for (dx, dy, dz) in zipped:
        part_translation = translation + torch.tensor([dx, dy, dz]).unsqueeze(0).to(device)
        contact_loss, collsion_loss = get_physical_loss(sdf, pcd, scale, part_translation, 
                                        move_limit=move_limit, move_axis=move_axis, move_origin=move_origin,
                                        move_type=move_type, move_samples=32,
                                        collision_margin=collision_margin, contact_margin=contact_margin)
        print(f'{contact_loss:.4f}, {collsion_loss:.4f}')
        if passed_physical_feasibility(contact_loss, collsion_loss, contact_thres, collision_thres):
            # make sure that at the position, the part cannot move further on x and y axis, (but allow move one step)
            if cat == 'slider_drawer':
                other_positions_to_check = [(i*grid_length, 0, 0) for i in range(1, step_thres+1)] + [(-i*grid_length, 0, 0) for i in range(1, step_thres+1)]
            elif cat == 'hinge_door':
                other_positions_to_check = [(0, 0, -i*grid_length) for i in range(1, step_thres+1)]
            else:
                other_positions_to_check = [(0, 0, i*grid_length) for i in range(1, step_thres+1)] + [(0, 0, -i*grid_length) for i in range(1, step_thres+1)]
            for ddx, ddy, ddz in other_positions_to_check:
                part_translation_dd = translation + torch.tensor([dx+ddx, dy+ddy, dz+ddz]).unsqueeze(0).to(device)
                contact_loss, collsion_loss = get_physical_loss(sdf, pcd, scale, part_translation_dd, 
                                            move_limit=move_limit, move_axis=move_axis, move_origin=move_origin,
                                            move_type=move_type, move_samples=32,
                                            collision_margin=collision_margin, contact_margin=contact_margin)
                if not passed_physical_feasibility(contact_loss, collsion_loss, contact_thres, collision_thres):
                    logger.log(f'Physical feasible at translation: {part_translation[0].cpu().numpy()}')
                    return 1
            logger.log(f'Physical feasible at translation: {part_translation[0].cpu().numpy()}' + \
                f' but also for all movement. Not physical feasible.')
            return 0
    logger.log(f'Not physical feasible for all translations.')
    return -1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('test_root', type=str) #  '../../part_meshes_recon/slider_drawer'
    parser.add_argument('--gt_root', type=str, default='/mnt/data-rundong/PartDiffusion/dataset')
    parser.add_argument('--test_list', type=str, default='../data_lists/test/')
    parser.add_argument('--cat', type=str, default='slider_drawer')
    parser.add_argument('--gpu_id', type=int, default=7)
    parser.add_argument('--contact_thres', type=float, default=0.04)
    parser.add_argument('--contact_margin', type=float, default=0.005)
    parser.add_argument('--collision_thres', type=float, default=0.04)
    parser.add_argument('--collision_margin', type=float, default=0.005)
    parser.add_argument('--grid_length', type=float, default=0.005)
    parser.add_argument('--step_thres', type=int, default=5)
    args = parser.parse_args()

    # slider_drawer: 0.08, 0.008, 0.08, 0.008 / 38 + 8 + 2
    # hinge door: 0.05, 0.005, 0.05, 0.005 / 50 + 5 + 5
    # hinge_knob : 0.05, 0.005, 0.05, 0.005 / 19 + 1 + 0
    # line_fixed: 0.05, 0.005, 0.05, 0.005 / 85 + 5 + 0

    device = f'cuda:{args.gpu_id}'

    args.test_root = os.path.abspath(args.test_root)
    logger = Logger('/'+os.path.join(*args.test_root.split('/')[:-1], 'physical_feasibility.txt'))
    logger.log('Contact threshold: {:.4f}'.format(args.contact_thres))
    logger.log('Collision threshold: {:.4f}'.format(args.collision_thres))
    logger.log('Contact margin: {:.4f}'.format(args.contact_margin))
    logger.log('Collision margin: {:.4f}'.format(args.collision_margin))

    good_objs = []
    all_not_feasible_objs = []
    multiple_feasible_objs = []

    test_obj_files = [x for x in os.listdir(args.test_root) if x.endswith('.obj')]
    with open(os.path.join(args.test_list, args.cat+'.txt'), 'r') as f:
        test_filenames = f.readlines()
    test_filenames = [f.strip() for f in test_filenames]
    test_obj_files = [f for f in test_obj_files if f.split('.')[0] in test_filenames]
    test_pcd_files = [f.replace('.obj', '.ply') for f in test_obj_files]

    test_obj_files = [os.path.join(args.test_root, f) for f in test_obj_files]
    test_pcd_files = [os.path.join(args.gt_root, 'part_ply_fps', args.cat, f) for f in test_pcd_files]
    
    if args.cat == 'slider_drawer':
        move_limit = (0, 0.1)
        move_type = 'translation'
        move_axis = [0, 0, 1]
        steps = (5, 5, 31)
    elif args.cat == 'hinge_door':
        move_limit = (0, 90)
        move_type = 'rotation'
        move_axis = [1, 0, 0]
        steps = (5, 5, 31)
    else:
        move_limit = move_type = move_axis = None
        steps = (5, 5, 21)

    for (obj_file, pcd_file) in zip(test_obj_files, test_pcd_files):
        # if not any([x in obj_file for x in ['10797_1.obj', '10068_2.obj', '10685_1.obj', '12038_1.obj']]):
        #     continue
        logger.log(f'Physical feasibility of {os.path.basename(obj_file)}')

        obj = trimesh.load(obj_file)
        # check watertight
        if not obj.is_watertight:
            logger.log(f'{os.path.basename(obj_file)} is not watertight, thus not physical feasible!')
            all_not_feasible_objs.append(os.path.basename(obj_file))
            continue
        splits = obj.split(only_watertight=True)
        splits = [split for split in splits if split.volume > 1e-7]
        if len(splits) != 1:
            logger.log(f'{os.path.basename(obj_file)} contains mutiple not connected parts, thus not physical feasible!')
            all_not_feasible_objs.append(os.path.basename(obj_file))
            continue
        obj = splits[0]
        # def is_inside(comp1, comp2):
        #     return np.all(comp2.contains(comp1.vertices))
        # def check_connectivity(splits):
        #     for i, comp1 in enumerate(splits):
        #         for comp2 in splits[i+1:]:
        #             if not is_inside(comp1, comp2) and not is_inside(comp2, comp1):
        #                 return 0
        #     return 1
        # if check_connectivity(splits) == 0:
        #     logger.log(f'{os.path.basename(obj_file)} contains mutiple not connected parts, thus not physical feasible!')
        #     continue

        pcd = open3d.io.read_point_cloud(pcd_file)
        pcd = torch.tensor(np.array(pcd.points), dtype=torch.float32).to(device) # (N, 3)
        
        result = physical_feasibility(obj, pcd, logger, device=device,
                                      grid_length=args.grid_length,
                                      step_thres=args.step_thres, steps=steps, 
                                      res=128, cat=args.cat,
                                      move_type=move_type, move_limit=move_limit, 
                                      move_axis=move_axis,
                                      collision_margin=args.collision_margin, contact_margin=args.contact_margin,
                                      contact_thres=args.contact_thres, collision_thres=args.collision_thres)
        if result == 1:
            good_objs.append(os.path.basename(obj_file))
        elif result == 0:
            multiple_feasible_objs.append(os.path.basename(obj_file))
        else:
            all_not_feasible_objs.append(os.path.basename(obj_file))
    

    logger.log(f'Good objects: {good_objs}')
    logger.log(f'Multiple feasible objects: {multiple_feasible_objs}')
    logger.log(f'All not feasible objects: {all_not_feasible_objs}')

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
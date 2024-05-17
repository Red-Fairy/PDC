import open3d
import os
from pytorch3d.loss import chamfer_distance
import torch
import numpy as np
import argparse
from tqdm import tqdm
import sys
sys.path.append('.')
from utils import AverageMeter
from utils_eval import calculate_fscore

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('test_meshes', type=str)
    parser.add_argument('gt_meshes', type=str) # default='../../part_meshes_recon/slider_drawer'
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--num_points', type=int, default=20000)
    args = parser.parse_args()

    device = f'cuda:{args.gpu_id}'

    # calculate chamfer distance and f-score
    log_path = os.path.join('/', *os.path.abspath(args.test_meshes).split('/')[:-1], 'metrics.txt')
    f = open(log_path, 'w')

    obj_files_test = sorted([os.path.join(args.test_meshes, f) for f in os.listdir(args.test_meshes) if f.endswith('.obj')])
    obj_files_gt = sorted([os.path.join(args.gt_meshes, f) for f in os.listdir(args.test_meshes) if f.endswith('.obj')])

    zipped = zip(obj_files_test, obj_files_gt)

    CUBE_SIDE_LEN = 2.0
    threshold_list = [CUBE_SIDE_LEN/500, CUBE_SIDE_LEN/400,
                      CUBE_SIDE_LEN/250, CUBE_SIDE_LEN/200, CUBE_SIDE_LEN/100]
    # threshold_list = [CUBE_SIDE_LEN/200]
    
    loss_meter_CD = AverageMeter()
    loss_fscore_meters = {th: AverageMeter() for th in threshold_list}

    for (obj_file_test, obj_file_gt) in tqdm(zipped):
        assert obj_file_test.split('/')[-1] == obj_file_gt.split('/')[-1]
        obj_test = open3d.io.read_triangle_mesh(obj_file_test)
        obj_test.translate(-obj_test.get_axis_aligned_bounding_box().get_center())

        obj_gt = open3d.io.read_triangle_mesh(obj_file_gt)
        obj_gt.translate(-obj_gt.get_axis_aligned_bounding_box().get_center())

        pcd_test = obj_test.sample_points_uniformly(number_of_points=args.num_points) 
        pcd_gt = obj_gt.sample_points_uniformly(number_of_points=args.num_points)

        pointclouds_test = torch.tensor(np.asarray(pcd_test.points)).to(device).unsqueeze(0) # (1, num_points, 3)
        pointclouds_gt = torch.tensor(np.asarray(pcd_gt.points)).to(device).unsqueeze(0) # (1, num_points, 3)
        
        loss = chamfer_distance(pointclouds_test, pointclouds_gt, batch_reduction='sum', point_reduction='mean')[0]
        loss_meter_CD.update(loss.item(), args.num_points)

        for th in threshold_list:
            f_score, precision, recall = calculate_fscore(pcd_test, pcd_gt, th=th)
            loss_fscore_meters[th].update(f_score, 1)

    f.write(f'Chamfer distance: {loss_meter_CD.avg}\n')
    for th in threshold_list:
        f.write(f'F-score_{th}: {loss_fscore_meters[th].avg}\n')

if __name__ == '__main__':
    main()

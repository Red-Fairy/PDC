import os
from tqdm import tqdm
import open3d
import argparse
import typing

def calculate_fscore(gt: open3d.geometry.PointCloud, pr: open3d.geometry.PointCloud, th: float=0.01) -> typing.Tuple[float, float, float]:
    '''Calculates the F-score between two point clouds with the corresponding threshold value.'''
    d1 = gt.compute_point_cloud_distance(pr)
    d2 = pr.compute_point_cloud_distance(gt)
    
    if len(d1) and len(d2):
        recall = float(sum(d < th for d in d2)) / float(len(d2))
        precision = float(sum(d < th for d in d1)) / float(len(d1))

        if recall+precision > 0:
            fscore = 2 * recall * precision / (recall + precision)
        else:
            fscore = 0
    else:
        fscore = 0
        precision = 0
        recall = 0

    return fscore, precision, recall

def main():
    CUBE_SIDE_LEN = 1.0

    open3d.utility.set_verbosity_level(open3d.utility.VerbosityLevel.Error)

    parser = argparse.ArgumentParser(description='F-score evaluation')
    parser.add_argument('pred_meshes', type=str)
    parser.add_argument('gt_meshes', type=str)

    parser.add_argument('--out_path', type=str, default=None)
    parser.add_argument('--th', type=float)

    parser.add_argument('--num_points', type=int, default=20000)

    args = parser.parse_args()

    out_path = os.path.basename(args.pred_meshes) if args.out_path is None else args.out_path
    print(out_path)
    os.makedirs(out_path, exist_ok=True)

    if args.th is None:
        threshold_list = [CUBE_SIDE_LEN/500, CUBE_SIDE_LEN/400,
                          CUBE_SIDE_LEN/250, CUBE_SIDE_LEN/200, CUBE_SIDE_LEN/100]
    else:
        threshold_list = [args.th]

    num_points = args.num_points

    model_names = [x for x in os.listdir(args.pred_meshes) if x.endswith(".obj")]

    result_dict = {}
    for th in threshold_list:
        result_dict[th] = {'fscore': 0, 'precision': 0, 'recall': 0}

    for model_name in tqdm(model_names):
        gt_mesh_path = os.path.join(args.gt_meshes, model_name)
        pr_mesh_path = os.path.join(args.pred_meshes, model_name)

        gt_mesh = open3d.io.read_triangle_mesh(gt_mesh_path)
        pr_mesh = open3d.io.read_triangle_mesh(pr_mesh_path)

        gt_pcd = gt_mesh.sample_points_uniformly(number_of_points=num_points)
        pr_pcd = pr_mesh.sample_points_uniformly(number_of_points=num_points)
        
        for th in threshold_list:
            f_score, precision, recall = calculate_fscore(gt_pcd, pr_pcd, th=th)
            result_dict[th]['fscore'] += f_score
            result_dict[th]['precision'] += precision
            result_dict[th]['recall'] += recall

    file_fscore, file_precision, file_recall = open(os.path.join(out_path, "fscore.txt"), "w"), open(os.path.join(out_path, "precision.txt"), "w"), open(os.path.join(out_path, "recall.txt"), "w")

    for th in threshold_list:
        result_dict[th]['fscore'] /= len(model_names)
        result_dict[th]['precision'] /= len(model_names)
        result_dict[th]['recall'] /= len(model_names)

        print(f"Threshold: {th}")
        print(f"F-score: {result_dict[th]['fscore']}")
        print(f"Precision: {result_dict[th]['precision']}")
        print(f"Recall: {result_dict[th]['recall']}")

        file_fscore.write(f"{result_dict[th]['fscore']}\n")
        file_precision.write(f"{result_dict[th]['precision']}\n")
        file_recall.write(f"{result_dict[th]['recall']}\n")

    file_fscore.close()
    file_precision.close()
    file_recall.close()

if __name__ == '__main__':
    main()

            
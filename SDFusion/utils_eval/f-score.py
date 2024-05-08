import os
import tqdm
import open3d
import argparse
import typing

def calculate_fscore(gt: open3d.geometry.PointCloud, pr: open3d.geometry.PointCloud, th: float=0.01) -> typing.Tuple[float, float, float]:
    '''Calculates the F-score between two point clouds with the corresponding threshold value.'''
    d1 = open3d.compute_point_cloud_to_point_cloud_distance(gt, pr)
    d2 = open3d.compute_point_cloud_to_point_cloud_distance(pr, gt)
    
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

CUBE_SIDE_LEN = 1.0

open3d.set_verbosity_level(open3d.utility.VerbosityLevel.Error)

parser = argparse.ArgumentParser(description='F-score evaluation')
parser.add_argument('--gt_models', type=str, required=True)
parser.add_argument('--pr_models', type=str, required=True)

parser.add_argument('--pr_path', type=str, required=True)
parser.add_argument('--out_path', type=str)
parser.add_argument('--th', type=float)

parser.add_argument('--num_points', type=int, default=10000)

args = parser.parse_args()

if args.out_path is None:
    out_path = "fscore"
else:
    out_path = args.out_path
os.mkdir(out_path)

if args.th is None:
    threshold_list = [CUBE_SIDE_LEN/200, CUBE_SIDE_LEN/100,
                      CUBE_SIDE_LEN/50, CUBE_SIDE_LEN/20,
                      CUBE_SIDE_LEN/10, CUBE_SIDE_LEN/5]
else:
    threshold_list = [args.th]

num_points = args.num_points

model_names = [x for x in os.listdir(args.pr_models) if x.endswith(".obj")]

result_dict = {}
for th in threshold_list:
    result_dict[th] = {'fscore': 0, 'precision': 0, 'recall': 0}

for model_name in tqdm(model_names):
    gt_mesh_path = os.path.join(args.gt_models, model_name)
    pr_mesh_path = os.path.join(args.pr_models, model_name)

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

        
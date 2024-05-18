import trimesh
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('root', type=str, help='root directory')
parser.add_argument('--mobility_filepath', type=str, 
                    default='/mnt/data-rundong/PartDiffusion/dataset/part_mobility/hinge_door',
                    help='mobility file path')
args = parser.parse_args()

dst_dir = os.path.join(**os.path.abspath(args.root).split('/')[:-1], args.root.split('/')[-1] + '_rotated')

obj_filenames = [os.path.join(args.root, f) for f in os.listdir(args.root) if f.endswith('.obj')]
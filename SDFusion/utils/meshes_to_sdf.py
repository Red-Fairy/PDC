import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import sys
sys.path.append('.')
from datasets import mesh_to_sdf
import h5py
import numpy as np
import trimesh
from termcolor import colored, cprint
import argparse
from tqdm import tqdm
import json
import skimage
from scipy.spatial.transform import Rotation as R

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='../../dataset/')
parser.add_argument('--cat', type=str, default='slider_drawer', help='category name')
parser.add_argument('--padding', type=float, default=0.2, help='length of the bounding box')
parser.add_argument('--res', type=int, default=128, help='resolution of the sdf')
parser.add_argument('--truncation', type=float, default=0.2, help='truncation of the sdf')
parser.add_argument('--gen_bbox', action='store_true', help='whether to generate the bounding box')
parser.add_argument('--gen_recon', action='store_true', help='whether to generate the reconstructed mesh')
parser.add_argument('--debug', action='store_true', help='whether to debug')
parser.add_argument('--rotation', action='store_true', help='whether to rotate the mesh')
parser.add_argument('--mirror', action='store_true', help='whether to mirror the mesh')
parser.add_argument('--scale', action='store_true', help='whether to scale the mesh')
parser.add_argument('--haoran_convention', action='store_true', help='whether to use haoran 3D convention')

args = parser.parse_args()

# CATS = ['slider_button', 'hinge_door', 'hinge_lid', 'line_fixed_handle', 
#         'hinge_handle', 'round_fixed_handle', 'slider_drawer', 'hinge_knob', 'slider_lid']
CATS = ['slider_drawer', 'hinge_door']
CATS = ['slider_button', 'hinge_lid', 'line_fixed_handle', 
        'hinge_handle', 'round_fixed_handle', 'hinge_knob', 'slider_lid']
# CATS = ['slider_button']

SPACING = 2. / args.res

ROTATION = args.rotation
MIRROR = args.mirror
SCALE = args.scale
RES = args.res

print(args)
suffix = ''

if __name__ == '__main__':
    for CAT in CATS:
        print('processing category:', CAT)
        mesh_basedir = os.path.join(args.root, 'full_meshes', CAT)
        sdf_basedir = os.path.join(args.root, f'full_sdf_{RES}', CAT)
        os.makedirs(sdf_basedir, exist_ok=True)

        if args.gen_recon:
            mesh_recon_basedir = os.path.join(args.root, f'full_meshes_recon', CAT)
            os.makedirs(mesh_recon_basedir, exist_ok=True)

        if args.gen_bbox:
            part_bbox_basedir = os.path.join(args.root, f'part_bbox_aligned', CAT)
            os.makedirs(part_bbox_basedir, exist_ok=True)

        filenames = [f for f in os.listdir(mesh_basedir) if f.endswith('.obj')]

        for file_name in tqdm(filenames):
            try:
                cprint(f'process mesh: {file_name}', 'green')

                object_id, part_id = file_name.split('.obj')[0].split('_')
                mesh = trimesh.load(os.path.join(mesh_basedir, file_name))

                # rotate the mesh by ZYX euler (90, 0, -90)
                if args.haoran_convention:
                    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(90), [0, 0, 1]))
                    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(0), [0, 1, 0]))
                    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0]))

                T = np.array(mesh.bounding_box.centroid)
                S = mesh.bounding_box.extents

                # generate the SDF
                sdf = mesh_to_sdf(mesh, args.res, padding=args.padding, trunc=args.truncation)

                if args.gen_recon:
                    vertices, faces, normals, _ = skimage.measure.marching_cubes(sdf.squeeze(0).cpu().numpy(), level=0.02, spacing=(SPACING, SPACING, SPACING))
                    mesh_recon = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
                    mesh_recon.apply_translation(-mesh_recon.bounding_box.centroid)
                    mesh_recon.apply_scale(np.max(S) / np.max(mesh_recon.bounding_box.extents))
                    mesh_recon.apply_translation(T)

                    recon_filename = os.path.join(mesh_recon_basedir, f'{object_id}_{part_id}.obj')
                    mesh_recon.export(recon_filename)

                sdf = sdf.reshape(-1, 1)
                h5_filename = file_name[:-4] + '.h5'
                h5f = h5py.File(os.path.join(sdf_basedir, h5_filename), 'w')
                h5f.create_dataset('pc_sdf_sample', data=sdf.cpu().numpy().astype(np.float32), compression='gzip', compression_opts=4)
                h5f.close()

                if args.gen_bbox:
                    bbox = {
                        'centroid': T.tolist(),
                        'extents': S.tolist()
                    }
                    with open(os.path.join(part_bbox_basedir, f'{object_id}_{part_id}.json'), 'w') as f:
                        json.dump(bbox, f)

                if args.debug:
                    break
            except:
                with open('error_log.txt', 'a') as f:
                    f.write(f'{file_name}\n')

            # if ROTATION:
            #     ## rotate the mesh
            #     # randomly rotate the mesh
            #     angle = np.random.uniform(0, 360)
            #     rotation_matrix = trimesh.transformations.rotation_matrix(
            #             angle=np.radians(angle),
            #             direction=[0, 0, 1], 
            #             point=mesh.centroid
            #         )
                
            #     mesh.apply_transform(rotation_matrix)
            #     bbox = np.dot(bbox, rotation_matrix[:3, :3].T)
            #     suffix += f'_rot{angle:.0f}'
            #     # apply the same rotation to the bbox

            # if MIRROR: 
            #     ## mirror the mesh
            #     mesh.apply_transform(trimesh.transformations.scale_matrix([1, 1, -1], mesh.centroid))
            #     bbox[:, 2] = -bbox[:, 2]
            #     suffix += '_mir'

            # if SCALE:
            #     ## scale the mesh
            #     scale = np.random.uniform(0.5, 1.5)
            #     mesh.apply_transform(trimesh.transformations.scale_matrix([scale, scale, scale], mesh.centroid))
            #     bbox = bbox * scale
            #     suffix += f'_scale{scale:.2f}'

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
from util_3d import mesh_to_sdf
import h5py
import numpy as np
import trimesh
from termcolor import colored, cprint
import argparse
from mesh_to_sdf import mesh_to_voxels
import skimage

parser = argparse.ArgumentParser()
parser.add_argument('--cat', type=str, default='slider_drawer', help='category name')
parser.add_argument('--length', type=float, default=1.2, help='length of the bounding box')
# parser.add_argument('--mesh_scale', type=float, default=1, help='scale of the mesh')
parser.add_argument('--res', type=int, default=64, help='resolution of the sdf')
parser.add_argument('--truncation', type=float, default=0.2, help='truncation of the sdf')
parser.add_argument('--rotation', action='store_true', help='whether to rotate the mesh')
parser.add_argument('--mirror', action='store_true', help='whether to mirror the mesh')
parser.add_argument('--scale', action='store_true', help='whether to scale the mesh')

args = parser.parse_args()

CAT = args.cat
LENGTH = args.length
SPACING = 2. / args.res

ROTATION = args.rotation
MIRROR = args.mirror
SCALE = args.scale

print(args)
suffix = ''

if __name__ == '__main__':
    mesh_basedir = f'/raid/haoran/Project/PartDiffusion/PartDiffusion/dataset/part_meshes/{CAT}'
    sdf_basedir = f'/raid/haoran/Project/PartDiffusion/PartDiffusion/dataset/part_sdf/{CAT}'
    if not os.path.exists(sdf_basedir):
        os.makedirs(sdf_basedir)

    filenames = [f for f in os.listdir(mesh_basedir) if f.endswith('.obj')]

    for file_name in filenames[:10]:

        mesh = trimesh.load(os.path.join(mesh_basedir, file_name))
        mesh.apply_translation(-mesh.centroid)
        mesh.apply_scale(1. / np.max(np.abs(mesh.bounds)))

        # create bbox (eight points) in the order (-x,+y,+z), (+x,+y,+z), (+x,-y,+z), (-x,-y,+z), (-x,+y,-z), (+x,+y,-z), (+x,-y,-z), (-x,-y,-z)
        # (x_min, y_min, z_min), (x_max, y_max, z_max) = mesh.bounds
        # bbox = np.array([[x_min, y_max, z_max], [x_max, y_max, z_max], [x_max, y_min, z_max], [x_min, y_min, z_max], [x_min, y_max, z_min], [x_max, y_max, z_min], [x_max, y_min, z_min], [x_min, y_min, z_min]])

        if ROTATION:
            ## rotate the mesh
            # randomly rotate the mesh
            angle = np.random.uniform(0, 360)
            rotation_matrix = trimesh.transformations.rotation_matrix(
                    angle=np.radians(angle),
                    direction=[0, 0, 1], 
                    point=mesh.centroid
                )
            
            mesh.apply_transform(rotation_matrix)
            bbox = np.dot(bbox, rotation_matrix[:3, :3].T)
            suffix += f'_rot{angle:.0f}'
            # apply the same rotation to the bbox

        if MIRROR: 
            ## mirror the mesh
            mesh.apply_transform(trimesh.transformations.scale_matrix([1, 1, -1], mesh.centroid))
            bbox[:, 2] = -bbox[:, 2]
            suffix += '_mir'

        if SCALE:
            ## scale the mesh
            scale = np.random.uniform(0.5, 1.5)
            mesh.apply_transform(trimesh.transformations.scale_matrix([scale, scale, scale], mesh.centroid))
            bbox = bbox * scale
            suffix += f'_scale{scale:.2f}'

        sdf = mesh_to_sdf(mesh, args.res, padding=args.length-1)

        # sdf to truncated sdf, thres=args.thres
        sdf = np.clip(sdf, -args.truncation, args.truncation)

        ''' save the mesh_recon '''
        vertices, faces, normals, _ = skimage.measure.marching_cubes(sdf, level=0.02, spacing=(SPACING, SPACING, SPACING))
        mesh_recon = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
        centroid = mesh_recon.centroid
        print(centroid)
        print(np.max(mesh_recon.bounding_box.extents))
        # mesh_recon.apply_translation(-centroid)
        # (x_min, y_min, z_min), (x_max, y_max, z_max) = mesh_recon.bounds
        # print(x_min, y_min, z_min, x_max, y_max, z_max)
        # recon_filename = file_name[:-4] + '_recon.obj' if suffix == '' else file_name[:-4] + '_recon{}.obj'.format(suffix)
        # mesh_recon.export(os.path.join(sdf_basedir, recon_filename))
        # print(mesh_recon.centroid)

        sdf = sdf.reshape(-1, 1)
        h5_filename = file_name[:-4] + '_sdf_res_64.h5'
        h5f = h5py.File(os.path.join(sdf_basedir, h5_filename), 'w')
        h5f.create_dataset('pc_sdf_sample', data=sdf.astype(np.float32), compression='gzip', compression_opts=4)
        h5f.close()
        cprint(f'process mesh: {file_name}', 'green')

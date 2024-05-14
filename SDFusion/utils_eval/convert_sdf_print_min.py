import sys
sys.path.append('.')
from utils import mesh_to_sdf, sdf_to_mesh_trimesh
import numpy as np
import trimesh

path = '/mnt/azureml/cr/j/18805bb336d249f6b409578dd210054f/exe/wd/data-rundong/PartDiffusion/dataset/part_meshes/slider_drawer/22339_0.obj'
mesh = trimesh.load_mesh(path)

sdf = mesh_to_sdf(mesh, padding=0.2, res=128, trunc=0.2)

print(sdf.min())

# back to trimesh
spacing = (2./128, 2./128, 2./128)
mesh = sdf_to_mesh_trimesh(sdf[0], level=0.015, spacing=spacing)

# then again to sdf
sdf = mesh_to_sdf(mesh, padding=0.2, res=128, trunc=0.2)
print(sdf.min())
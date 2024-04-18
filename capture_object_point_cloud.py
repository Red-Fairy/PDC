import sapien.core as sapien
from sapien.utils.viewer import Viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import trimesh
import glob
import os
from tqdm import tqdm
USE_VIEWER = False

save_root = '/raid/haoran/Project/PartDiffusion/PartDiffusion/dataset/part_ply'

categories = ['line_fixed_handle', 'hinge_handle', 'hinge_knob', 'hinge_lid', 'round_fixed_handle', 'slider_button']

for category in categories:

    object_names = [x for x in os.listdir(os.path.join('/raid/haoran/Project/PartDiffusion/PartDiffusion/dataset/part_meshes', category)) if x.endswith('.obj')]

    obj_part_tuple = [(x.split('_')[0], x.split('_')[-1].split('.')[0]) for x in object_names]
    obj_part_tuple = sorted(obj_part_tuple, key=lambda x: x[0])

    for i, (obj_id, part_id) in tqdm(enumerate(obj_part_tuple)):
        print(obj_id, part_id)
        path = os.path.join('/raid/haoran/Project/data/partnet_all_annotated_new/annotation', obj_id, f'remove_single_part_wo-{category}-link_{part_id}.urdf')
        assert os.path.exists(path)

        # Initialize the Sapien Engine
        engine = sapien.Engine()
        renderer = sapien.VulkanRenderer()
        engine.set_renderer(renderer)

        # Create a scene
        scene = engine.create_scene()
        scene.set_timestep(1 / 240.0)
        scene.set_ambient_light([0.5, 0.5, 0.5])
        scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
        scene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)
        scene.add_point_light([1, -2, 2], [1, 1, 1], shadow=True)
        scene.add_point_light([-1, 0, 1], [1, 1, 1], shadow=True)
        if USE_VIEWER:
            viewer = Viewer(renderer)
            viewer.set_scene(scene)
            viewer.set_camera_xyz(x=1.2, y=0.25, z=0.4)
            viewer.set_camera_rpy(r=0, p=0, y=0)
        scene.add_ground(-5)

        # Load URDF
        loader = scene.create_urdf_loader()
        urdf_obj = loader.load(path)
        urdf_obj.set_pose(sapien.Pose([0, 0, 0], [0, 1, 0, 0]))
        # Assuming the object is static or its pose has been set already
        pose = urdf_obj.get_pose()
        print(pose)
        options = [[4, 0.00, 1, 0, -20, 0]
            , [0., 2, 1, -90, -40, 0]
            , [0., -2, 1, 90, -40, 0],
            [-4, 0.00, 1, 0, 200, 0],
            [4, 0.00, -1, 0, 40, 0]
            , [0., 2, -1, -90, 40, 0]
            , [0., -2, -1, 90, 40, 0],
            [-4, 0.00, -1, 0, -200, 0],
            [0, 0.00, 3, 0, 270, 0],
            [0, 0.00, -3, 0, 90, 0]]
        cameras = [
            scene.add_camera(
                name="camera",
                width=640,
                height=480,
                fovy=np.deg2rad(35),
                near=0.1,
                far=100,
            ) for cam in options
        ]

        for cam_id, cam_pose in enumerate(options):
            q = R.from_euler('xyz', cam_pose[3:], degrees=True).as_quat()
            cameras[cam_id].set_pose(sapien.Pose(p=cam_pose[:3], q=q))    

        scene.update_render()
        [cam.take_picture() for cam in cameras]
        positions = [cam.get_float_texture('Position') for cam in cameras]
        model_matrixs = [cam.get_model_matrix() for cam in cameras]
        points_worlds = []
        for cam_id in range(len(cameras)):
            position = positions[cam_id]
            model_matrix = model_matrixs[cam_id]
            points_opengl = position.reshape(-1, position.shape[-1])[..., :3]
            points_world = points_opengl @ model_matrix[:3, :3].T + model_matrix[:3, 3]
            points_worlds.append(points_world)
        points_worlds = np.concatenate(points_worlds, axis=0)
        valid_mask = (points_worlds[:,2]<=2.9)&(points_worlds[:,2]>=-2.9)\
        &(points_worlds[:,1]<=1.9)&(points_worlds[:,1]>=-1.9)\
        &(points_worlds[:,0]<=3.9)&(points_worlds[:,0]>=-3.9)
        points_worlds = points_worlds[valid_mask]
        # rgbas = [cam.get_float_texture('Color') for cam in cameras]
        # points_colors = [rgba.reshape(-1, rgba.shape[-1])[..., :3] for rgba in rgbas]
        # points_colors = np.concatenate(points_colors, axis=0)[valid_mask]
        # rgba_imgs = [(rgba * 255).clip(0, 255).astype("uint8") for rgba in rgbas]

        # due to different camera conventions, we need to rotate the point cloud
        # rotate around x axis by 180 degree, y and z by 90 degree
        rotation = R.from_euler('xyz', [180, 90, 90], degrees=True)
        # rotation_y = R.from_euler('xyz', [0, 90, 0], degrees=True)
        # rotation_z = R.from_euler('xyz', [0, 0, 90], degrees=True)
        points_worlds = points_worlds @ rotation.as_matrix().T

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_worlds)
        # pcd.colors = o3d.utility.Vector3dVector(points_colors) # we do not use colors
        
        p_name  = path.split("/")[-1].split(".")[0]
        # import pdb; pdb.set_trace()
        save_path = os.path.join(save_root, category, f"{obj_id}_{part_id}.ply")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        o3d.io.write_point_cloud(save_path, pcd)

        print(len(points_worlds))

        scene = None
        engine = None


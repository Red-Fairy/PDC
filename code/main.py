from structure.image import ObjIns
from structure.image import save_point_cloud_to_ply
import glob, os, sys, json, pickle
from PIL import Image
import numpy as np

PART_ID2NAME = {
    0: 'others'             ,
    1: 'line_fixed_handle'  ,
    2: 'round_fixed_handle' ,
    3: 'slider_button'      ,
    4: 'hinge_door'         ,
    5: 'slider_drawer'      ,
    6: 'slider_lid'         ,
    7: 'hinge_lid'          ,
    8: 'hinge_knob'         ,
    9: 'revolute_handle'    ,
}

IMG_ROOT = "./remove_path_render"
rgb_paths = glob.glob(f"{IMG_ROOT}/rgb/Sto*original.png")
TARGET_PART = "slider_drawer"
SAVE_ROOT = f"output_{TARGET_PART}"
os.makedirs(SAVE_ROOT, exist_ok=True)
def read_gapartnet_files(root, name):
    rgb_path = f"{root}/rgb/{name}.png"
    depth_path = f"{root}/depth/{name}.npz"
    meta_path = f"{root}/metafile/{name}.json"
    npcs_path = f"{root}/npcs/{name}.npz"
    seg_path = f"{root}/segmentation/{name}.npz"
    bbox_path = f"{root}/bbox/{name}.pkl"
    
    rgb_map = np.array(Image.open(rgb_path))
    depth_map = np.load(depth_path)['depth_map']
    meta_dict = json.load(open(meta_path))
    npcs_map = np.load(npcs_path)['npcs_map']
    sem_seg_map = np.load(seg_path)['semantic_segmentation']
    ins_seg_map = np.load(seg_path)['instance_segmentation']
    bbox = pickle.load(open(bbox_path, 'rb'))

    bbox_pose_dict = bbox['bbox_pose_dict']
    parts_sem_ids = []
    parts_ins_ids = []
    parts_bboxes = []
    parts_link_names = []
    for link_name in bbox_pose_dict:
        bbox_info = bbox_pose_dict[link_name]
        parts_sem_ids.append(bbox_info['category_id'])
        parts_ins_ids.append(bbox_info['instance_id'])
        parts_bboxes.append(bbox_info['bbox'])
        parts_link_names.append(link_name)
    
    return rgb_map, depth_map, meta_dict, npcs_map, sem_seg_map, ins_seg_map, \
        parts_sem_ids, parts_ins_ids, parts_bboxes, parts_link_names

total = len(rgb_paths)
i_ =0 
for rgb_path in rgb_paths[:]:
    i_+=1
    
    name = rgb_path.split("/")[-1].split(".")[0]
    name_remove = name[:-9]
    print(i_, total, name_remove)
    if os.path.exists(f"{SAVE_ROOT}/{name_remove}_data.npy"):
        print("already exists, continue!")
        continue
    
    rgb_map, depth_map, meta_dict, npcs_map, sem_seg_map, ins_seg_map, parts_sem_ids, \
        parts_ins_ids, parts_bboxes, parts_link_names = read_gapartnet_files(IMG_ROOT, name)

    img_obj_orig = ObjIns(
        name = name,
        cate = name.split("_")[0],
        image = rgb_map,
        depth = depth_map,
        K = np.array(meta_dict['camera_intrinsic']).reshape(3, 3),
        world2camera_rotation = np.array(meta_dict['world2camera_rotation']).reshape(3, 3),
        camera2world_translation = np.array(meta_dict['camera2world_translation']),
        image_reso = (rgb_map.shape[0], rgb_map.shape[1]),
        img_sem_map = sem_seg_map,
        img_ins_map = ins_seg_map,
        img_npcs_map = npcs_map,
        parts_sem_ids=parts_sem_ids,
        parts_ins_ids=parts_ins_ids,
        parts_link_names=parts_link_names,
        parts_bboxes=parts_bboxes,
    )
    # rgb to gbr
    # img_obj.pcs_rgb = img_obj.pcs_rgb[:, [2, 1, 0]]

    name = name_remove
    rgb_map, depth_map, meta_dict, npcs_map, sem_seg_map, ins_seg_map, parts_sem_ids, \
        parts_ins_ids, parts_bboxes, parts_link_names = read_gapartnet_files(IMG_ROOT, name)

    img_obj_remove = ObjIns(
        name = name,
        cate = name.split("_")[0],
        image = rgb_map,
        depth = depth_map,
        K = np.array(meta_dict['camera_intrinsic']).reshape(3, 3),
        world2camera_rotation = np.array(meta_dict['world2camera_rotation']).reshape(3, 3),
        camera2world_translation = np.array(meta_dict['camera2world_translation']),
        image_reso = (rgb_map.shape[0], rgb_map.shape[1]),
        img_sem_map = sem_seg_map,
        img_ins_map = ins_seg_map,
        img_npcs_map = npcs_map,
        parts_sem_ids=parts_sem_ids,
        parts_ins_ids=parts_ins_ids,
        parts_link_names=parts_link_names,
        parts_bboxes=parts_bboxes,
    )
    
    complement_link = [(index, item) for index, item in enumerate(img_obj_orig.parts_link_names) if item not in img_obj_remove.parts_link_names]
    complement_bbox = [img_obj_orig.parts_bboxes[index] for index, item in complement_link] # using name and parts_bboxes
    complement_sem = [img_obj_orig.parts_sem_ids[index] for index, item in complement_link] 
    complement_sem_name = [PART_ID2NAME[item+1] for item in complement_sem]
    complement_ins = [img_obj_orig.parts_ins_ids[index] for index, item in complement_link]
    if TARGET_PART not in complement_sem_name:
        print(complement_sem_name, "not in, continue!")
        continue

    img_obj_orig.get_pc()
    img_obj_orig.get_downsampled_pc()
    img_obj_remove.get_pc()
    img_obj_remove.get_downsampled_pc()
    
    
    data = {
        "complement_link": complement_link,
        "complement_bbox": complement_bbox,
        "orig_pcs_downsample": img_obj_orig.pcs_xyz,
        "orig_pcs_rgb_downsample": img_obj_orig.pcs_rgb,
        "remove_pcs_downsample": img_obj_remove.pcs_xyz,
        "remove_pcs_rgb_downsample": img_obj_remove.pcs_rgb,
        "orig_img": img_obj_orig.image,
        "remove_img": img_obj_remove.image,
        "orig_img_sem": img_obj_orig.img_sem_map,
        "orig_img_ins": img_obj_orig.img_ins_map,
        "remove_img_sem": img_obj_remove.img_sem_map,
        "remove_img_ins": img_obj_remove.img_ins_map,
    }
    np.save(f"{SAVE_ROOT}/{name_remove}_data.npy", data, allow_pickle=True)
    img_obj_orig.visualization(
        SAVE_ROOT,
        options=["img", "img_gt_ins", "img_gt_sem", "img_gt_bbox"],
        render_text=False,
    )
    img_obj_remove.visualization(
        SAVE_ROOT,
        options=["img", "img_gt_ins", "img_gt_sem", "img_gt_bbox"],
        render_text=False,
    )
    save_point_cloud_to_ply(img_obj_remove.pcs_xyz, img_obj_remove.pcs_rgb[:, [2, 1, 0]]*255, f"{SAVE_ROOT}/{name}.ply")
    save_point_cloud_to_ply(img_obj_orig.pcs_xyz, img_obj_orig.pcs_rgb[:, [2, 1, 0]]*255, f"{SAVE_ROOT}/{name}.ply")

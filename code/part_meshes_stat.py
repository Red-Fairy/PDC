import os, glob

cates = ['slider_drawer', 'hinge_door', 'line_fixed_handle', 'round_fixed_handle', 'slider_button', 'slider_lid', 'hinge_lid', 'hinge_knob', 'hinge_handle']
for cate in cates:
    paths = glob.glob(f"/data/haoran/Projects/GAPartNet_docs/part_meshes/{cate}/*.obj")
    print(cate, len(paths))

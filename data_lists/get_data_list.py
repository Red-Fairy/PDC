import os
import numpy as np

root_dir = '/raid/haoran/Project/PartDiffusion/PartDiffusion/dataset/part_ply'

list_dir = '/raid/haoran/Project/PartDiffusion/PartDiffusion/data_lists'

# cats = os.listdir(root_dir)
# cats = [cat for cat in cats if os.path.isdir(os.path.join(root_dir, cat))]

cats = ['hinge_door', 'slider_drawer']
for cat in cats:
    os.makedirs(os.path.join(list_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(list_dir, 'test'), exist_ok=True)

test_ratio = 0.1

for cat in cats:
    files = os.listdir(os.path.join(root_dir, cat))
    files = [x for x in files if int(x.split('/')[-1].split('_')[0]) > 99999]
    print(files)

    np.random.shuffle(files)
    test_files = files[:int(len(files) * test_ratio)]
    train_files = files[int(len(files) * test_ratio):]

    with open(os.path.join(list_dir, 'test', cat+'.txt'), 'a') as f:
        for file in test_files:
            f.write(file.split('.')[0] + '\n')

    with open(os.path.join(list_dir, 'train', cat+'.txt'), 'a') as f:
        for file in train_files:
            f.write(file.split('.')[0] + '\n')



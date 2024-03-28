import os
import numpy as np

root_dir = '/raid/haoran/Project/PartDiffusion/PartDiffusion/dataset/part_sdf_64'

list_dir = '/raid/haoran/Project/PartDiffusion/PartDiffusion/data_lists'

cats = os.listdir(root_dir)

test_ratio = 0.1

for cat in cats:
    files = os.listdir(os.path.join(root_dir, cat))

    np.random.shuffle(files)
    test_files = files[:int(len(files) * test_ratio)]
    train_files = files[int(len(files) * test_ratio):]

    with open(os.path.join(list_dir, 'test', cat+'.txt'), 'w') as f:
        for file in test_files:
            f.write(file.split('.')[0] + '\n')

    with open(os.path.join(list_dir, 'train', cat+'.txt'), 'w') as f:
        for file in train_files:
            f.write(file.split('.')[0] + '\n')



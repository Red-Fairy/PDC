import json
import os

info = {}
basedir = '/mnt/seagate12t/GAPartNet/test/PartSDF.normalized'
cat_list = ['button', 'door', 'drawer', 'handle', 'knob', 'lid']
for cat in cat_list:
    h5file_list = []
    catdir = os.path.join(basedir, cat)
    for file_name in os.listdir(catdir):
        if file_name.endswith('.h5'):
            h5file_list.append(file_name)
    info[cat] = h5file_list
print(f'len(h5file_list): {len(h5file_list)}')
print(f'h5file_list: {h5file_list}')

with open(f'{basedir}/info.json', 'w') as f:
    json.dump(info, f)

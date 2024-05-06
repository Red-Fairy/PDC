import json
import os

root = '../ignore_files/slider_drawer/set_0'

files = os.listdir(root)

sum = 0
count = 0

for file in files:
    json_file = open(os.path.join(root, file))
    data = json.load(json_file)
    angle_diff = abs(data['rotate_angle'] - data['rotate_angle_pred'])
    sum += angle_diff
    count += 1

print(sum / count)
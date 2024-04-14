import os

root = '/raid/haoran/Project/PartDiffusion/PartDiffusion/SDFusion/logs/slider-ply2shape-plyrot-scale3-lr0.00001/test_200000_rotate{:.1f}_scale3.0_eta0.0_steps50/log.txt'

obj_loss_dict = {}

for angle in range(0, 360, 10):
    log_path = root.format(angle)
    with open(log_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            obj_id = line.split(',')[0].split(' ')[-1]
            loss = line.split(',')[1].strip()
            if obj_id not in obj_loss_dict:
                obj_loss_dict[obj_id] = {angle: float(loss)}
            else:
                obj_loss_dict[obj_id][angle] = float(loss)

for obj_id, obj_losses in obj_loss_dict.items():

    # print max and average
    max_loss = max(obj_losses.values())
    avg_loss = sum(obj_losses.values()) / len(obj_losses)
    print("Object ID: ", obj_id,
        "Max Loss: ", max_loss,
        "Average Loss: ", avg_loss)
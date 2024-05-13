logs_dir='logs'

### hyper params ###

### model stuff ###
model='cvae-ply2shape'
cvae_cfg='configs/cvae-ply2shape-128.yaml'

cond_ckpt="../pretrained_checkpoint/pointnet2.pth"

### dataset stuff ###
max_dataset_size=1000000
dataset_mode='gapnet'
dataroot="/home/puhao/data/PartDiff/raid/haoran/Project/PartDiffusion/PartDiffusion/dataset/"

res=128
trunc_thres=0.2
#####################

### display & log stuff ###
display_freq=1
print_freq=25
total_iters=250000
save_steps_freq=25000
###########################

multi_gpu=0  # multi-gpu
batch_size=2
name=$1
cat=$2
lr=$3
port=$4
gpu_ids="0"

name="${name}-cvae-scale${uc_scale}-lr${lr}"

args="--name ${name} --logs_dir ${logs_dir} --gpu_ids ${gpu_ids} \
            --lr ${lr} --batch_size ${batch_size} --max_dataset_size ${max_dataset_size} \
            --model ${model} --cvae_cfg ${cvae_cfg} \
            --dataset_mode ${dataset_mode} --res ${res} --cat ${cat} --trunc_thres ${trunc_thres} \
            --display_freq ${display_freq} --print_freq ${print_freq} \
            --total_iters ${total_iters} --save_steps_freq ${save_steps_freq} \
            --dataroot ${dataroot} \
            --ply_cond \
            --cond_ckpt ${cond_ckpt}"
            ## --continue_train --load_iter 200000"

echo "[*] Training is starting on `hostname`, GPU#: ${gpu_ids}, logs_dir: ${logs_dir}"

# set available gpus
if [ $multi_gpu = 1 ]; then
    accelerate launch --multi_gpu --gpu_ids $gpu_ids --num_processes 2 \
         --main_process_port $port --mixed_precision 'no' train_accelerate.py $args
else
    # accelerate launch --multi_gpu --gpu_ids $gpu_ids --num_processes 2 \
    #      --main_process_port $port --mixed_precision 'no' train_accelerate.py $args
    python train.py $args

fi

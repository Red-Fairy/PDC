logs_dir='logs'

### hyper params ###

### model stuff ###
model='cvae-ply2shape'
cvae_cfg='configs/cvae-ply2shape-128.yaml'

cond_ckpt="/mnt/data-rundong/PartDiffusion/pretrained_checkpoint/pointnet2.pth"

### dataset stuff ###
max_dataset_size=1000000
dataset_mode='gapnet'
dataroot="/mnt/data-rundong/PartDiffusion/dataset/"

res=128
trunc_thres=0.2
#####################

### display & log stuff ###
display_freq=50
print_freq=250
total_iters=50000
save_steps_freq=5000
###########################

multi_gpu=1  # multi-gpu
batch_size=4
cat=$1
lr=$2
port=$3
gpu_ids=$4

name="cvae_${cat}"

args="--name ${name} --logs_dir ${logs_dir} --gpu_ids ${gpu_ids} \
            --lr ${lr} --batch_size ${batch_size} --max_dataset_size ${max_dataset_size} \
            --model ${model} --cvae_cfg ${cvae_cfg} \
            --dataset_mode ${dataset_mode} --res ${res} --cat ${cat} --trunc_thres ${trunc_thres} \
            --display_freq ${display_freq} --print_freq ${print_freq} \
            --total_iters ${total_iters} --save_steps_freq ${save_steps_freq} \
            --dataroot ${dataroot} \
            --ply_cond --ply_rotate \
            --cond_ckpt ${cond_ckpt} \
            --continue_train "

echo "[*] Training is starting on `hostname`, GPU#: ${gpu_ids}, logs_dir: ${logs_dir}"

# set available gpus
if [ $multi_gpu = 1 ]; then
    accelerate launch --multi_gpu --gpu_ids $gpu_ids --num_processes 4 \
         --main_process_port $port --mixed_precision 'no' train_accelerate.py $args
else
    python train.py $args
fi

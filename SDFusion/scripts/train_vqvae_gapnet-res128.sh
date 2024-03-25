SLURM=0

RED='\033[0;31m'
NC='\033[0m' # No Color
DATE_WITH_TIME=`date "+%Y-%m-%dT%H-%M-%S"`

logs_dir='logs'

### set gpus ###
gpu_ids=0          # single-gpu
# gpu_ids=0,1,2,3  # multi-gpu

if [ ${#gpu_ids} -gt 1 ]; then
    # specify these two if multi-gpu
    # NGPU=2
    # NGPU=3
    NGPU=4
    PORT=11768
    echo "HERE"
fi
################


### model stuff ###
model="vqvae"
vq_cfg="configs/vqvae_gapnet-128.yaml"
####################


### dataset stuff ###
max_dataset_size=10000000
dataset_mode='gapnet'
dataroot="/raid/haoran/Project/PartDiffusion/PartDiffusion/dataset/"

###########################

### display & log stuff ###
display_freq=1000 # default: log every display_freq batches
print_freq=100 # default:
total_iters=500000
save_steps_freq=50000
###########################

### hyper params ###
multi_gpu=1

batch_size=1
res=128
trunc_thres=0.2
name=$1
lr=$2
port=$3
gpu_ids=$4
cat="all"

name="${name}-vqvae-lr${lr}"

args="--name ${name} --logs_dir ${logs_dir} --gpu_ids ${gpu_ids} --lr ${lr} --batch_size ${batch_size} \
                --model ${model} --vq_cfg ${vq_cfg} \
                --dataroot ${dataroot} --dataset_mode ${dataset_mode} --cat ${cat} \
                --res ${res} --trunc_thres ${trunc_thres} --max_dataset_size ${max_dataset_size} \
                --display_freq ${display_freq} --print_freq ${print_freq} \
                --total_iters ${total_iters} --save_steps_freq ${save_steps_freq} 
                --continue_train --joint_rotate "

echo "[*] Training is starting on `hostname`, GPU#: ${gpu_ids}, logs_dir: ${logs_dir}"

if [ $multi_gpu = 1 ]; then
    accelerate launch --multi_gpu --num_processes 2 --gpu_ids $gpu_ids --main_process_port $port --mixed_precision 'no' train_accelerate.py $args
else
    python train.py $args
fi

CAT=$1
BATCHSIZE=$2
RELEASE_NOTE=$3
multi_gpu=1
gpu_ids=6,7  # multi-gpu
SLURM=1

RED='\033[0;31m'
NC='\033[0m' # No Color
DATE_WITH_TIME=`date "+%Y-%m-%dT%H-%M-%S"`

logs_dir='logs'

### hyper params ###
lr=1e-4
batch_size=8

### model stuff ###
model='sdfusion'
df_cfg='configs/sdfusion_snet.yaml'

vq_model="vqvae"
vq_dset='gapnet'
vq_cat="slider_drawer"
vq_ckpt="/raid/haoran/Project/PartDiffusion/PartDiffusion/pretrained_checkpoint/vqvae-snet-all.pth"
vq_cfg="/raid/haoran/Project/PartDiffusion/PartDiffusion/SDFusion/configs/vqvae_snet.yaml"

### dataset stuff ###
max_dataset_size=1000000
dataset_mode='gapnet'
dataroot="/raid/haoran/Project/PartDiffusion/PartDiffusion/dataset/part_sdf/slider_drawer"

res=64
cat="slider_drawer"
trunc_thres=0.2
#####################

### display & log stuff ###
display_freq=250
print_freq=25
total_iters=500000
save_steps_freq=5000
###########################

data_version=v.2.0

today=$(date '+%m%d')
me=`basename "$0"`
me=$(echo $me | cut -d'.' -f 1)

note=$RELEASE_NOTE

debug=0
if [ $debug = 1 ]; then
    printf "${RED}Debugging!${NC}\n"
	batch_size=1
    # batch_size=40
	max_dataset_size=120
    save_steps_freq=3
	display_freq=2
	print_freq=2
    name="DEBUG-${name}"
fi

name="drawer-uncond-test"
load_iter="10000"

args="--name ${name} --logs_dir ${logs_dir} --gpu_ids ${gpu_ids} \
            --lr ${lr} --batch_size ${batch_size} --max_dataset_size ${max_dataset_size} \
            --model ${model} --df_cfg ${df_cfg} \
            --vq_model ${vq_model} --vq_cfg ${vq_cfg} --vq_ckpt ${vq_ckpt} --vq_dset ${vq_dset} --vq_cat ${vq_cat} \
            --dataset_mode ${dataset_mode} --res ${res} --cat ${cat} --trunc_thres ${trunc_thres} \
            --display_freq ${display_freq} --print_freq ${print_freq} \
            --total_iters ${total_iters} --save_steps_freq ${save_steps_freq} \
            --debug ${debug} --dataroot ${dataroot} --data_version ${data_version}"

# set available gpus
if [ $multi_gpu = 1 ]; then
    accelerate launch --multi_gpu --gpu_ids $gpu_ids --main_process_port 29512 --mixed_precision 'no' train_accelerate.py $args
else
    python train_accelerate.py $args
fi


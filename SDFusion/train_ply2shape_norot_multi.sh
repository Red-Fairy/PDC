multi_gpu=1  # multi-gpu
SLURM=1

RED='\033[0;31m'
NC='\033[0m' # No Color
DATE_WITH_TIME=`date "+%Y-%m-%dT%H-%M-%S"`

logs_dir='logs'

### hyper params ###

### model stuff ###
model='sdfusion-ply2shape'
df_cfg='configs/sdfusion-ply2shape.yaml'

vq_model="vqvae"
vq_dset='gapnet'
vq_cat="slider_drawer"
vq_ckpt="/raid/haoran/Project/PartDiffusion/PartDiffusion/pretrained_checkpoint/vqvae-snet-all.pth"
vq_cfg="/raid/haoran/Project/PartDiffusion/PartDiffusion/SDFusion/configs/vqvae_snet.yaml"

cond_ckpt="/raid/haoran/Project/PartDiffusion/PartDiffusion/pretrained_checkpoint/pointnet2.pth"

### dataset stuff ###
max_dataset_size=1000000
dataset_mode='gapnet'
dataroot="/raid/haoran/Project/PartDiffusion/PartDiffusion/dataset/part_sdf"
cat="slider_drawer"

res=64
trunc_thres=0.2
#####################

### display & log stuff ###
display_freq=250
print_freq=25
total_iters=250000
save_steps_freq=2500
###########################

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

batch_size=12
name=$1
lr=$2
port=$3
gpu_ids=$4
uc_scale=$5

name="${name}-norot-scale${uc_scale}-lr${lr}"


args="--name ${name} --logs_dir ${logs_dir} --gpu_ids ${gpu_ids} \
            --lr ${lr} --batch_size ${batch_size} --max_dataset_size ${max_dataset_size} \
            --model ${model} --df_cfg ${df_cfg} \
            --vq_model ${vq_model} --vq_cfg ${vq_cfg} --vq_ckpt ${vq_ckpt} --vq_dset ${vq_dset} --vq_cat ${vq_cat} \
            --dataset_mode ${dataset_mode} --res ${res} --cat ${cat} --trunc_thres ${trunc_thres} \
            --display_freq ${display_freq} --print_freq ${print_freq} \
            --total_iters ${total_iters} --save_steps_freq ${save_steps_freq} \
            --debug ${debug} --dataroot ${dataroot} \
            --ply_cond --cond_ckpt ${cond_ckpt} --uc_scale ${uc_scale} \
            --continue_train "

echo "[*] Training is starting on `hostname`, GPU#: ${gpu_ids}, logs_dir: ${logs_dir}"

# set available gpus
if [ $multi_gpu = 1 ]; then
    accelerate launch --multi_gpu --gpu_ids $gpu_ids --main_process_port $port --mixed_precision 'no' train_accelerate.py $args
else
    python train.py $args
fi




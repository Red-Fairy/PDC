
CAT=$1
RELEASE_NOTE=$2
SLURM=0

RED='\033[0;31m'
NC='\033[0m' # No Color
DATE_WITH_TIME=`date "+%Y-%m-%dT%H-%M-%S"`

logs_dir='logs_home'
### set gpus ###
gpu_ids=3          # single-gpu
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

### hyper params ###
lr=1e-4
batch_size=8
if [ $SLURM = 1 ]; then
    batch_size=16
fi
####################

### model stuff ###
model='sdfusion'
df_cfg='configs/sdfusion_snet.yaml'

vq_model="vqvae"
vq_cfg=None
vq_ckpt=None
if [ $CAT = "handle" ]; then
    vq_cfg="logs_home/slurms/2023-08-25T16-13-32-vqvae-gapnet-handle-res64-LR1e-4-T0.2-2A100@16bs/vqvae_gapnet.yaml"
    vq_ckpt="logs_home/slurms/2023-08-25T16-13-32-vqvae-gapnet-handle-res64-LR1e-4-T0.2-2A100@16bs/ckpt/vqvae_epoch-best.pth"
fi
if [ $CAT = "knob" ]; then
    vq_cfg=logs_home/slurms/2023-08-27T18-21-18-vqvae-gapnet-knob-res64-LR1e-4-T0.2-2A100@16bs/vqvae_gapnet.yaml
    vq_ckpt=logs_home/slurms/2023-08-27T18-21-18-vqvae-gapnet-knob-res64-LR1e-4-T0.2-2A100@16bs/ckpt/vqvae_epoch-best.pth
fi
vq_dset='gapnet'
vq_cat=${CAT}
if [ $SLURM = 1 ]; then
    if [ $CAT = "handle" ]; then
        vq_cfg="logs_home/2023-08-25T16-13-32-vqvae-gapnet-handle-res64-LR1e-4-T0.2-2A100@16bs/vqvae_gapnet.yaml"
        vq_ckpt="logs_home/2023-08-25T16-13-32-vqvae-gapnet-handle-res64-LR1e-4-T0.2-2A100@16bs/ckpt/vqvae_epoch-best.pth"
    fi
fi

####################

### dataset stuff ###
max_dataset_size=10000000
dataset_mode='gapnet'
dataroot="/mnt/seagate12t/GAPartNet/test/PartSDF.normalized"
if [ $SLURM = 1 ]; then
    dataroot="/scratch/generalvision/GAPartNet/PartSDF.normalized"
fi
res=64
cat=${CAT}
trunc_thres=0.2
#####################

### display & log stuff ###
display_freq=500
print_freq=25
total_iters=100000000
save_steps_freq=3000
###########################


today=$(date '+%m%d')
me=`basename "$0"`
me=$(echo $me | cut -d'.' -f 1)

note=$RELEASE_NOTE

name="${DATE_WITH_TIME}-${model}-${dataset_mode}-${cat}-LR${lr}-${note}"

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

cmd="train.py --name ${name} --logs_dir ${logs_dir} --gpu_ids ${gpu_ids} \
            --lr ${lr} --batch_size ${batch_size} --max_dataset_size ${max_dataset_size} \
            --model ${model} --df_cfg ${df_cfg} \
            --vq_model ${vq_model} --vq_cfg ${vq_cfg} --vq_ckpt ${vq_ckpt} --vq_dset ${vq_dset} --vq_cat ${vq_cat} \
            --dataset_mode ${dataset_mode} --res ${res} --cat ${cat} --trunc_thres ${trunc_thres} \
            --display_freq ${display_freq} --print_freq ${print_freq}
            --total_iters ${total_iters} --save_steps_freq ${save_steps_freq} \
            --debug ${debug}"

if [ ! -z "$dataroot" ]; then
    cmd="${cmd} --dataroot ${dataroot}"
    echo "setting dataroot to: ${dataroot}"
fi

if [ ! -z "$ckpt" ]; then
    cmd="${cmd} --ckpt ${ckpt}"
    echo "continue training with ckpt=${ckpt}"
fi

multi_gpu=0
if [ ${#gpu_ids} -gt 1 ]; then
    multi_gpu=1
fi

echo "[*] Training is starting on `hostname`, GPU#: ${gpu_ids}, logs_dir: ${logs_dir}"

if [ $multi_gpu = 1 ]; then
    cmd="-m torch.distributed.launch --nproc_per_node=${NGPU} --master_port=${PORT} ${cmd}"
fi

echo "[*] Training with command: "
echo "[*] CUDA_VISIBLE_DEVICES=${gpu_ids} python ${cmd}"

# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=${gpu_ids} python ${cmd}
CUDA_VISIBLE_DEVICES=${gpu_ids} python ${cmd}
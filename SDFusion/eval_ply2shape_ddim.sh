RED='\033[0;31m'
NC='\033[0m' # No Color
DATE_WITH_TIME=`date "+%Y-%m-%dT%H-%M-%S"`

logs_dir='logs'

### hyper params ###
batch_size=8
name=$1
gpu_ids=$2

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
dataroot="/raid/haoran/Project/PartDiffusion/PartDiffusion/dataset/part_sdf/slider_drawer"

res=64
cat="slider_drawer"
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

args="--name ${name} --logs_dir ${logs_dir} --gpu_ids ${gpu_ids} \
            --batch_size ${batch_size} --max_dataset_size ${max_dataset_size} \
            --model ${model} --df_cfg ${df_cfg} \
            --vq_model ${vq_model} --vq_cfg ${vq_cfg} --vq_ckpt ${vq_ckpt} --vq_dset ${vq_dset} --vq_cat ${vq_cat} \
            --dataset_mode ${dataset_mode} --res ${res} --cat ${cat} --trunc_thres ${trunc_thres} \
            --total_iters ${total_iters} \
            --debug ${debug} --dataroot ${dataroot} \
            --ply_cond --cond_ckpt ${cond_ckpt}"

CUDA_VISIBLE_DEVICES=$gpu_ids python test_new.py $args




logs_dir='logs'

### hyper params ###

### model stuff ###
model='sdfusion-plybbox2shape'
df_cfg='configs/sdfusion-plybbox2shape.yaml'

vq_model="vqvae"
vq_dset='gapnet'
vq_cat="slider_drawer"
vq_ckpt="logs/gapnet-res128-vqvae-lr0.00002/ckpt/vqvae_steps-latest.pth"
vq_cfg="configs/vqvae_gapnet-128.yaml"

cond_ckpt="../pretrained_checkpoint/pointnet2.pth"

### dataset stuff ###
max_dataset_size=1000000
dataset_mode='gapnet'
dataroot="../../data-rundong/PartDiffusion/dataset/"

res=128
trunc_thres=0.2
#####################

### display & log stuff ###
display_freq=250
print_freq=25
total_iters=75000
save_steps_freq=5000
###########################

multi_gpu=1
batch_size=4
name=$1
cat=$2
lr=$3
port=$4
gpu_ids=$5
uc_ply_scale=$6
uc_bbox_scale=$7

name="${name}-plybbox2shape-norot-scale${uc_scale}-lr${lr}"


args="--name ${name} --logs_dir ${logs_dir} --gpu_ids ${gpu_ids} \
            --lr ${lr} --batch_size ${batch_size} --max_dataset_size ${max_dataset_size} \
            --model ${model} --df_cfg ${df_cfg} \
            --vq_model ${vq_model} --vq_cfg ${vq_cfg} --vq_ckpt ${vq_ckpt} --vq_dset ${vq_dset} --vq_cat ${vq_cat} \
            --dataset_mode ${dataset_mode} --res ${res} --cat ${cat} --trunc_thres ${trunc_thres} \
            --display_freq ${display_freq} --print_freq ${print_freq} \
            --total_iters ${total_iters} --save_steps_freq ${save_steps_freq} \
            --dataroot ${dataroot} \
            --ply_rotate --ply_bbox_cond \
            --cond_ckpt ${cond_ckpt} --uc_ply_scale ${uc_ply_scale} --uc_bbox_scale ${uc_bbox_scale} "

echo "[*] Training is starting on `hostname`, GPU#: ${gpu_ids}, logs_dir: ${logs_dir}"

# set available gpus
if [ $multi_gpu = 1 ]; then
    accelerate launch --multi_gpu --gpu_ids $gpu_ids --main_process_port $port --mixed_precision 'no' train_accelerate.py $args
else
    python train.py $args
fi




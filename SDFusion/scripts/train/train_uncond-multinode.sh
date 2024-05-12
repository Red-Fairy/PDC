### model stuff ###
model='sdfusion'
df_cfg='configs/sdfusion-uncond-128.yaml'

vq_model="vqvae"
vq_dset='gapnet'
vq_cat="slider_drawer"
vq_ckpt="/mnt/data-rundong/PartDiffusion/SDFusion/logs/full-part-gen-vqvae-lr0.00001/ckpt/vqvae_steps-latest.pth"
vq_cfg="configs/vqvae_gapnet-128.yaml"

cond_ckpt="/mnt/data-rundong/PartDiffusion/pretrained_checkpoint/pointnet2.pth"

### dataset stuff ###
max_dataset_size=1000000
dataset_mode='gapnet'
dataroot="/mnt/data-rundong/PartDiffusion/dataset/"

res=128
trunc_thres=0.2
#####################

### display & log stuff ###
display_freq=250
print_freq=50
total_iters=100000
save_steps_freq=5000
###########################

multi_gpu=1
batch_size=5
sdf_mode='full'
name=$1
cat=$2
lr=$3
nnode=$4

logs_dir='/mnt/data-rundong/PartDiffusion/SDFusion/logs'
name="${name}-uncond-lr${lr}"

args="--name ${name} --logs_dir ${logs_dir} \
            --lr ${lr} --batch_size ${batch_size} --max_dataset_size ${max_dataset_size} \
            --model ${model} --df_cfg ${df_cfg} \
            --vq_model ${vq_model} --vq_cfg ${vq_cfg} --vq_ckpt ${vq_ckpt} --vq_dset ${vq_dset} --vq_cat ${vq_cat} \
            --dataset_mode ${dataset_mode} --res ${res} --cat ${cat} --trunc_thres ${trunc_thres} \
            --display_freq ${display_freq} --print_freq ${print_freq} \
            --total_iters ${total_iters} --save_steps_freq ${save_steps_freq} \
            --dataroot ${dataroot} --sdf_mode ${sdf_mode}"

# set available gpus
if [ $multi_gpu = 1 ]; then
    torchrun --nproc_per_node=8 --nnode=${nnode} --node_rank=$AZUREML_CR_NODE_RANK --master_addr=$AZ_BATCHAI_JOB_MASTER_NODE_IP --master_port=9901 train_accelerate.py $args
else
    python train.py $args
fi




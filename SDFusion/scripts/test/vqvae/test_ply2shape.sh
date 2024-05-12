### model stuff ###
model="vqvae"
vq_cfg="configs/vqvae_gapnet-128.yaml"
####################

### dataset stuff ###
max_dataset_size=10000000
dataset_mode='gapnet'
dataroot="/mnt/data-rundong/PartDiffusion/dataset/"

res=128
trunc_thres=0.2
#####################

### hyper params ###
logs_dir='logs'
cat='slider_drawer'
batch_size=4
name=$1
gpu_ids=$2
load_iter=$3

# --loss_margin 0.00390625 1/256

args="--name ${name} --logs_dir ${logs_dir} --gpu_ids ${gpu_ids} \
            --batch_size ${batch_size} --max_dataset_size ${max_dataset_size} \
            --model ${model} --vq_cfg ${vq_cfg} \
            --dataroot ${dataroot} --cat ${cat} \
            --dataset_mode ${dataset_mode} --sdf_mode full \
            --res ${res} --trunc_thres ${trunc_thres} \
            --load_iter ${load_iter} "

CUDA_VISIBLE_DEVICES=$gpu_ids python test_vq.py $args


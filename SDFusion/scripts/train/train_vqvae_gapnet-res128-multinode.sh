### model stuff ###
model="vqvae"
vq_cfg="configs/vqvae_gapnet-128.yaml"
####################

### dataset stuff ###
max_dataset_size=10000000
dataset_mode='gapnet'
dataroot="/mnt/data-rundong/PartDiffusion/dataset/"

###########################

### display & log stuff ###
display_freq=500 # default: log every display_freq batches
print_freq=50 # default:
total_iters=75000
save_steps_freq=20000
###########################

### hyper params ###
multi_gpu=1

batch_size=1
res=128
trunc_thres=0.2
name=$1
lr=$2
cat="all"

logs_dir="/mnt/data-rundong/PartDiffusion/SDFusion/logs"
name="${name}-vqvae-lr${lr}"

args="--name ${name} --logs_dir ${logs_dir} --lr ${lr} --batch_size ${batch_size} \
                --model ${model} --vq_cfg ${vq_cfg} --sdf_mode full \
                --dataroot ${dataroot} --dataset_mode ${dataset_mode} --cat ${cat} \
                --res ${res} --trunc_thres ${trunc_thres} --max_dataset_size ${max_dataset_size} \
                --display_freq ${display_freq} --print_freq ${print_freq} \
                --total_iters ${total_iters} --save_steps_freq ${save_steps_freq} 
                --ply_rotate --continue_train "

if [ $multi_gpu = 1 ]; then
    torchrun --nproc_per_node=8 --nnode=1 --node_rank=$AZUREML_CR_NODE_RANK --master_addr=$AZ_BATCHAI_JOB_MASTER_NODE_IP --master_port=9901 train_accelerate.py $args
else
    python train.py $args
fi

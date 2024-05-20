logs_dir='logs'

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
display_freq=250
print_freq=25
total_iters=250000
save_steps_freq=2500
###########################

### hyper params ###
batch_size=4
name=$1
gpu_ids=$2
load_iter=$3
cat=$4
testset_idx=0
# test_description="margin0.005_set${testset_idx}"
test_description="margin0.005_demo"

# 0.00390625 1/256
# 0.0078125 1/128

args="--name ${name} --logs_dir ${logs_dir} --gpu_ids ${gpu_ids} \
            --batch_size ${batch_size} --max_dataset_size ${max_dataset_size} \
            --model ${model} --cvae_cfg ${cvae_cfg} \
            --dataset_mode ${dataset_mode} --res ${res} --cat ${cat} --trunc_thres ${trunc_thres} \
            --total_iters ${total_iters} --dataroot ${dataroot} \
            --ply_rotate \
            --haoran \
            --ply_cond --cond_ckpt ${cond_ckpt} --load_iter ${load_iter} \
            --ddim_steps 50 --uc_scale 3 \
            --loss_margin 0.005 \
            --test_description ${test_description} --testset_idx ${testset_idx} \
            --use_mobility_constraint --model_id 19898"

CUDA_VISIBLE_DEVICES=$gpu_ids python test.py $args


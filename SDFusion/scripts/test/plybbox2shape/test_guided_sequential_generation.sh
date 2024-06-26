logs_dir='logs'

### model stuff ###
model='sdfusion-plybbox2shape'
df_cfg='configs/sdfusion-plybbox2shape.yaml'

vq_model="vqvae"
vq_dset='gapnet'
vq_cat="slider_drawer"
vq_ckpt="/mnt/data-rundong/PartDiffusion/SDFusion/logs/gapnet-res128-vqvae-lr0.00002/ckpt/vqvae_steps-latest.pth"
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
print_freq=25
total_iters=250000
save_steps_freq=2500
###########################

### hyper params ###
batch_size=1
name=$1
gpu_ids=$2
cat=$3
test_description=$4
model_id="19898_15"
# 19898: 1 door & 6 drawers
# 25493: 3 drawers
# 19179: 2 drawers
# 45238: 2 doors & 2 drawers
# 41083: 2 doors & 2 drawers (similar to 45238)
# 45749: 2 doors & 2 drawers
# 45146: 1 door & 1 drawer

uc_scale=3
load_iter=200000

# slider-ply2shape-plyrot-scale3-lr0.00001

# --loss_margin 0.00390625 1/256
# --loss_margin 0.0078125 1/128

args="--name ${name} --logs_dir ${logs_dir} --gpu_ids ${gpu_ids} \
            --batch_size ${batch_size} --max_dataset_size ${max_dataset_size} \
            --model ${model} --df_cfg ${df_cfg} \
            --vq_model ${vq_model} --vq_cfg ${vq_cfg} --vq_ckpt ${vq_ckpt} --vq_dset ${vq_dset} --vq_cat ${vq_cat} \
            --dataset_mode ${dataset_mode} --res ${res} --cat ${cat} --trunc_thres ${trunc_thres} \
            --total_iters ${total_iters} --dataroot ${dataroot} \
            --use_mobility_constraint \
            --guided_inference \
            --loss_margin 0.005 \
            --test_diversity --pcd_dir part_ply_rm_multi \
            --ply_bbox_cond --cond_ckpt ${cond_ckpt} \
            --uc_scale ${uc_scale} --load_iter ${load_iter} --model_id ${model_id} \
            --ddim_steps 50 --test_description ${test_description}"

CUDA_VISIBLE_DEVICES=$gpu_ids python test.py $args 


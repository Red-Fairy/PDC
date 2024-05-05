logs_dir='logs'

### model stuff ###
model='sdfusion-ply2shape'
df_cfg='configs/sdfusion-ply2shape-128.yaml'

vq_model="vqvae"
vq_dset='gapnet'
vq_cat="slider_drawer"
vq_ckpt="logs/gapnet-res128-vqvae-lr0.00002/ckpt/vqvae_steps-latest.pth"
vq_cfg="configs/vqvae_gapnet-128.yaml"

cond_ckpt="../pretrained_checkpoint/pointnet2.pth"

### dataset stuff ###
max_dataset_size=1000000
dataset_mode='gapnet'
dataroot="/mnt/azureml/cr/j/19c62471467141d39f5f0dc988c1ea42/exe/wd/data-rundong/PartDiffusion/dataset"

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
load_iter=$3
model_id='20411_0'
cat="slider_drawer"
mobility_type="rotation"
# cat="hinge_door"
# mobility_type="rotation"
rotate_angle=$4

# slider-ply2shape-plyrot-scale3-lr0.00001

# --loss_margin 0.00390625 1/256
# --loss_margin 0.0078125 1/128

args="--name ${name} --logs_dir ${logs_dir} --gpu_ids ${gpu_ids} \
            --batch_size ${batch_size} --max_dataset_size ${max_dataset_size} \
            --model ${model} --df_cfg ${df_cfg} \
            --vq_model ${vq_model} --vq_cfg ${vq_cfg} --vq_ckpt ${vq_ckpt} --vq_dset ${vq_dset} --vq_cat ${vq_cat} \
            --dataset_mode ${dataset_mode} --res ${res} --cat ${cat} --trunc_thres ${trunc_thres} \
            --total_iters ${total_iters} --dataroot ${dataroot} \
            --ply_rotate \
            --mobility_type ${mobility_type} \
            --use_mobility_constraint \
            --rotate_angle ${rotate_angle} \
            --scale_mode volume \
            --guided_inference \
            --haoran \
            --loss_margin 0.00390625 \
            --test_diversity \
            --ply_cond --cond_ckpt ${cond_ckpt} --load_iter ${load_iter} \
            --start_idx $5 --end_idx $6 \
            --ddim_steps 50 --uc_scale 3 --test_description margin128_haoran "

CUDA_VISIBLE_DEVICES=$gpu_ids python test.py $args &


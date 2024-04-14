experiment_name=$1
gpu_ids="0"
port=29500
multi_gpu=0
args="--ply_rotate \
    --load_pretrain \
    --experiment_name $experiment_name \
    --extend_size_train 5000 \
    --extend_size_test 500"

if [ $multi_gpu = 1 ]; then
    accelerate launch --multi_gpu --gpu_ids $gpu_ids --num_processes 2 \
         --main_process_port $port --mixed_precision 'no' train_scale_prediction.py $args
else
    python train_scale_prediction.py $args
fi

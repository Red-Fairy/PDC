# PartDiffusion
Physic-plausible Part Generation

## Installation
Package required: accelerate + diffusers + other necessary packages (listed in requirements.txt) + pytorch3d + kaolin

Install pytorch (cuda version 12.1)
```pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1```

```pip install -r requirements.txt```

Install pytorch3d
```pip install "git+https://github.com/facebookresearch/pytorch3d.git"```

Install kaolin (adjust the command based on the pytorch and cuda version)
```pip install kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.1_cu121.html``` 


## Usage

scripts are stored under `SDFusion/scripts` folder.
Run train_ply2shape_multi.sh for multi-gpu training (powered by accelerate package)
Run train_ply2shape_multi_norot.sh for multi-gpu training (powered by accelerate package)

GaPartNet dataset is implemented in `datasets/gapnet_dataset.py`

`--joint_rot` is used to jointly rotate input part and input point cloud, 

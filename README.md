# PartDiffusion
Physic-plausible Part Generation

## Installation
Package required: accelerate + diffusers + SDFusion requirements

```pip install accelerate diffusers```

```pip install -r requirements.txt```

## Usage

scripts are stored under `SDFusion/scripts` folder.
Run train_ply2shape_multi.sh for multi-gpu training (powered by accelerate package)
Run train_ply2shape_multi_norot.sh for multi-gpu training (powered by accelerate package)

GaPartNet dataset is implemented in `datasets/gapnet_dataset.py`

`--joint_rot` is used to jointly rotate input part and input point cloud, 

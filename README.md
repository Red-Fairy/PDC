# PartDiffusion
Physic-plausible Part Generation

## Installation
Package required for SDFusion + Accelerate + Diffusers + some other small packages (all can be installed by pip)

## Usage

scripts are stored under `SDFusion/scripts` folder.
Run train_ply2shape_multi.sh for multi-gpu training (powered by accelerate package)
Run train_ply2shape_multi_norot.sh for multi-gpu training (powered by accelerate package)

GaPartNet dataset is implemented in `datasets/gapnet_dataset.py`

`--joint_rot` is used to jointly rotate input part and input point cloud, 

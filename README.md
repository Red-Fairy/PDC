# PartDiffusion
Physic-plausible Part Generation

## Installation
Package required: accelerate + diffusers + other necessary packages (listed in requirements.txt) + pytorch3d + kaolin

Install pytorch following instructions on pytorch.org

Install pytorch3d

```pip install scikit-image matplotlib imageio plotly opencv-python```

```pip install black usort flake8 flake8-bugbear flake8-comprehensions```

We recomment install pytorch3d from source
```pip install "git+https://github.com/facebookresearch/pytorch3d.git"```

Install kaolin (adjust the command based on the pytorch and cuda version)

```pip install kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.1_cu117.html``` 

Install other required packages

```pip install -r requirements.txt```


## Usage

scripts are stored under `SDFusion/scripts` folder.
Run train_ply2shape_multi.sh for multi-gpu training (powered by accelerate package)

GaPartNet dataset is implemented in `datasets/gapnet_dataset.py`


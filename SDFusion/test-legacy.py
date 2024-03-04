import os
import time
import inspect

from termcolor import colored, cprint
from tqdm import tqdm

import torch.backends.cudnn as cudnn
# cudnn.benchmark = True

from options.test_options import TestOptions
from datasets.dataloader import CreateDataLoader, get_data_generator
from models.base_model import create_model

from utils.distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)

import torch
import random
import numpy as np
print(f'CUDA TORCH AVAILABLE: {torch.cuda.is_available()}')
from utils.visualizer import Visualizer
from utils.util_3d import sdf_to_mesh, sdf_to_mesh_trimesh


def test_main_worker(opt, model, visualizer, device):
    if get_rank() == 0:
        cprint('[*] Start inference. name: %s' % opt.name, 'blue')

    iter_start_time = time.time()
    for iter_i in range(opt.total_iters):
        cprint(f'[*] iter_i: {iter_i} . {opt.total_iters}', 'blue')

        opt.iter_i = iter_i
        iter_ip1 = iter_i + 1

        if get_rank() == 0:
            visualizer.reset()
        
        gen_df, intermediates = model.uncond(ngen=opt.batch_size)
        x_inter, pred_x0, timesteps = intermediates.values()

        ## create tqdm progress bar
        pbar = tqdm(total=len(timesteps), desc=f'Verbose Log', )
        for i_t, i_pred_x0 in zip(timesteps, pred_x0):
            # i_pred_x0_sdf = model.vqvae_module.decode_no_quant(i_pred_x0)
            i_pred_x0_sdf = model.vqvae_module.decode(i_pred_x0)
            for i_shape in range(opt.batch_size):
                os.makedirs(f'{opt.results_dir}/{opt.name}/samples.verbose/{iter_i}_{i_shape}', exist_ok=True)
                sdf = i_pred_x0_sdf[i_shape].detach().squeeze(0).cpu().numpy()
                mesh = sdf_to_mesh_trimesh(sdf, level=0.02)
                mesh.export(f'{opt.results_dir}/{opt.name}/samples.verbose/{iter_i}_{i_shape}/{iter_i}_{i_shape}@{i_t}.obj')
            pbar.update(1)
        
        os.makedirs(f'{opt.results_dir}/{opt.name}/samples', exist_ok=True)
        for i_shape in range(opt.batch_size):
            sdf = gen_df[i_shape].squeeze(0).cpu().numpy()

            #* add constraint here
            # sdf[:50, :, :16] = 0.1
            # sdf[:50, :, -16:] = 0.1

            mesh = sdf_to_mesh_trimesh(sdf, level=0.02)
            mesh.export(f'{opt.results_dir}/{opt.name}/samples/{iter_i}_{i_shape}.obj')


if __name__ == '__main__':
    ## set random seed
    torch.backends.cudnn.benchmark = False     
    torch.backends.cudnn.deterministic = True
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # this will parse args, setup log_dirs, multi-gpus
    opt = TestOptions().parse_and_setup()
    device = opt.device
    rank = opt.rank

    # get current time, print at terminal. easier to track exp
    from datetime import datetime
    opt.exp_time = datetime.now().strftime('%Y-%m-%dT%H-%M')

    # create model with loading ckpt
    model = create_model(opt)
    cprint(f'[*] "{opt.model}" initialized.', 'cyan')

    # visualizer
    visualizer = Visualizer(opt)
    if get_rank() == 0:
        visualizer.setup_io()

    # save model and dataset files
    if get_rank() == 0:
        expr_dir = '%s/%s' % (opt.results_dir, opt.name)
        model_f = inspect.getfile(model.__class__)
        cprint(f'[*] saving ckpt folder name: {opt.ckpt}', 'blue')
        modelf_out = os.path.join(expr_dir, os.path.basename(model_f))
        os.system(f'cp {model_f} {modelf_out}')

        if opt.vq_cfg is not None:
            vq_cfg = opt.vq_cfg
            cfg_out = os.path.join(expr_dir, os.path.basename(vq_cfg))
            os.system(f'cp {vq_cfg} {cfg_out}')
            
        if opt.df_cfg is not None:
            df_cfg = opt.df_cfg
            cfg_out = os.path.join(expr_dir, os.path.basename(df_cfg))
            os.system(f'cp {df_cfg} {cfg_out}')
        
    test_main_worker(opt, model, visualizer, device)

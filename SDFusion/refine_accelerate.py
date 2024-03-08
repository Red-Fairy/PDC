'''
Author: Rundong Luo
'''

import os
import time
import inspect

from termcolor import colored, cprint
from tqdm import tqdm

import torch.backends.cudnn as cudnn
# cudnn.benchmark = True

from options.refine_options import RefineOptions
from datasets.dataloader import CreateDataLoader, get_data_generator
from models.base_model import create_model
from accelerate import Accelerator

import torch
import random
import numpy as np

from utils.visualizer import Visualizer

from datasets.gapnet_utils import get_single_model

torch.autograd.set_detect_anomaly(True)

def train_main_worker(opt, model, accelerator: Accelerator):

    if accelerator.is_main_process:
        # setup visualizer for the main process
        visualizer = Visualizer(opt)
        visualizer.setup_io()
        # start training
        cprint('[*] Start training. name: %s' % opt.name, 'blue')

    pbar = tqdm(total=opt.total_iters, disable=not accelerator.is_local_main_process)
    pbar.set_description("Training Iters")
    # pbar = tqdm(total=opt.total_iters)

    iter_start_time = time.time()
    for iter_i in range(1, opt.total_iters+1):

        # opt.iter_i = iter_i
        iter_ip1 = iter_i + 1

        if accelerator.is_main_process:
            visualizer.reset()

        model.optimize_parameters(iter_i)

        if accelerator.is_main_process:

            if iter_i % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_errors(iter_i, errors, t)

            # display every n batches
            if iter_i % opt.display_freq == 0:
                # eval
                visualizer.display_current_results(model.get_current_visuals(), iter_i, phase='train')

            if iter_ip1 % opt.save_latest_freq == 0:
                cprint('saving the latest model (current_iter %d)' % (iter_i), 'blue')
                latest_name = f'steps-latest'
                model.save(latest_name, iter_ip1)

            # save every 3000 steps (batches)
            if iter_ip1 % opt.save_steps_freq == 0:
                cprint('saving the model at iters %d' % iter_ip1, 'blue')
                latest_name = f'steps-latest'
                model.save(latest_name, iter_ip1)
                cur_name = f'steps-{iter_ip1}'
                model.save(cur_name, iter_ip1)

        pbar.update(1)
        

if __name__ == "__main__":
    ## set random seed
    torch.backends.cudnn.benchmark = False     
    torch.backends.cudnn.deterministic = True
    # seed = 42
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)

    # create accelerator
    accelerator = Accelerator()

    # this will parse args, setup log_dirs, multi-gpus
    opt = RefineOptions().parse_and_setup(accelerator)
    # device = opt.device
    # rank = opt.rank

    # CUDA_VISIBLE_DEVICES = int(os.environ["LOCAL_RANK"]) 
    # import pdb; pdb.set_trace()

    # get current time, print at terminal. easier to track exp
    from datetime import datetime
    opt.exp_time = datetime.now().strftime('%Y-%m-%dT%H-%M')

    # read data
    input_instance = get_single_model(opt)

    # main loop
    model = create_model(opt, accelerator, input_instance)
    cprint(f'[*] "{opt.model}" initialized.', 'cyan')

    # save model and dataset files
    if accelerator.is_main_process:
        expr_dir = '%s/%s' % (opt.logs_dir, opt.name)
        model_f = inspect.getfile(model.__class__)
        cprint(f'[*] saving model files: {model_f}', 'blue')
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

    train_main_worker(opt, model, accelerator)

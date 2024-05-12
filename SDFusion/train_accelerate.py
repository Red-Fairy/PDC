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

from options.train_options import TrainOptions
from datasets.dataloader import CreateDataLoader, get_data_generator
from models.base_model import create_model
from accelerate import Accelerator

import torch
import random
import numpy as np

from utils.visualizer import Visualizer

torch.autograd.set_detect_anomaly(True)

def train_main_worker(opt, model, train_dl, test_dl, accelerator: Accelerator):

    if accelerator.is_main_process:
        # setup visualizer for the main process
        visualizer = Visualizer(opt)
        visualizer.setup_io()
        # start training
        cprint('[*] Start training. name: %s' % opt.name, 'blue')

    train_dg = get_data_generator(train_dl)
    test_dg = get_data_generator(test_dl)

    train_dg, test_dg = accelerator.prepare(train_dg, test_dg)

    pbar = tqdm(total=opt.total_iters, disable=not accelerator.is_main_process)
    pbar.update(model.start_iter)
    pbar.set_description("Training Iters")
    # pbar = tqdm(total=opt.total_iters)

    iter_start_time = time.time()
    for iter_i in range(model.start_iter+1, opt.total_iters+1):

        # opt.iter_i = iter_i
        iter_ip1 = iter_i + 1

        if accelerator.is_main_process:
            visualizer.reset()
        
        data = next(train_dg)
        if iter_i == 0 and accelerator.is_main_process:
            print(f"data Shape on single GPU: {data['sdf'].shape}")

        model.set_input(data)
        model.optimize_parameters(iter_i)

        if accelerator.is_main_process:

            if iter_i % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = time.time() - iter_start_time
                visualizer.print_current_errors(iter_i, errors, t)

            # display every n batches
            if iter_i % opt.display_freq == 0:
                # eval
                model.inference(data)
                visualizer.display_current_results(model.get_current_visuals(), iter_i, phase='train')

                test_data = next(test_dg)
                model.inference(test_data)
                visualizer.display_current_results(model.get_current_visuals(), iter_i, phase='test')

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

        # model.update_learning_rate()

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
    opt = TrainOptions().parse_and_setup(accelerator)
    # device = opt.device
    # rank = opt.rank

    # CUDA_VISIBLE_DEVICES = int(os.environ["LOCAL_RANK"]) 
    # import pdb; pdb.set_trace()

    # get current time, print at terminal. easier to track exp
    from datetime import datetime
    opt.exp_time = datetime.now().strftime('%Y-%m-%dT%H-%M')

    train_dl, test_dl = CreateDataLoader(opt)
    train_ds, test_ds = train_dl.dataset, test_dl.dataset

    dataset_size = len(train_ds)
    if opt.dataset_mode == 'shapenet_lang':
        cprint('[*] # training text snippets = %d' % len(train_ds), 'yellow')
        cprint('[*] # testing text snippets = %d' % len(test_ds), 'yellow')
    else:
        cprint('[*] # training images = %d' % len(train_ds), 'yellow')
        cprint('[*] # testing images = %d' % len(test_ds), 'yellow')

    # main loop
    model = create_model(opt, accelerator)
    cprint(f'[*] "{opt.model}" initialized.', 'cyan')

    # save model and dataset files
    if accelerator.is_main_process:
        expr_dir = '%s/%s' % (opt.logs_dir, opt.name)
        model_f = inspect.getfile(model.__class__)
        dset_f = inspect.getfile(train_ds.__class__)
        cprint(f'[*] saving model and dataset files: {model_f}, {dset_f}', 'blue')
        modelf_out = os.path.join(expr_dir, os.path.basename(model_f))
        dsetf_out = os.path.join(expr_dir, os.path.basename(dset_f))
        os.system(f'cp {model_f} {modelf_out}')
        os.system(f'cp {dset_f} {dsetf_out}')

        if opt.vq_cfg is not None:
            vq_cfg = opt.vq_cfg
            cfg_out = os.path.join(expr_dir, os.path.basename(vq_cfg))
            os.system(f'cp {vq_cfg} {cfg_out}')
            
        if opt.df_cfg is not None:
            df_cfg = opt.df_cfg
            cfg_out = os.path.join(expr_dir, os.path.basename(df_cfg))
            os.system(f'cp {df_cfg} {cfg_out}')

    train_main_worker(opt, model, train_dl, test_dl, accelerator)

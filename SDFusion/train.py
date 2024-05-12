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

from utils.visualizer import Visualizer

cuda_avail = torch.cuda.is_available()
# import pdb; pdb.set_trace()
print(f"CUDA TORCH AVAILABLE: {cuda_avail}")


def train_main_worker(opt, model, train_dl, test_dl, ):

    if get_rank() == 0:
        # setup visualizer for the main process
        visualizer = Visualizer(opt)
        visualizer.setup_io()
        cprint('[*] Start training. name: %s' % opt.name, 'blue')

    train_dg = get_data_generator(train_dl)
    test_dg = get_data_generator(test_dl)

    pbar = tqdm(total=opt.total_iters)
    pbar.update(model.start_iter)

    iter_start_time = time.time()
    for iter_i in range(model.start_iter+1, opt.total_iters+1):

        opt.iter_i = iter_i
        iter_ip1 = iter_i + 1

        if get_rank() == 0:
            visualizer.reset()
        
        data = next(train_dg)
        if iter_i == 0 and get_rank() == 0:
            print(f"!!!! data Shape on single GPU: {data['sdf'].shape}")
        model.set_input(data)
        model.optimize_parameters(iter_i)

        if get_rank() == 0:
            if iter_i % opt.print_freq == 0:
                errors = model.get_current_errors()

                t = (time.time() - iter_start_time) / opt.batch_size
                epoch_steps = iter_i
                visualizer.print_current_errors(iter_i, errors, t)

            # display every n batches
            if iter_i % opt.display_freq == 0:
                if iter_i == 0 and opt.debug == "1":
                    pbar.update(1)
                    continue

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

            # eval every 3000 steps
            if iter_ip1 % opt.save_steps_freq == 0:
                metrics = model.eval_metrics(test_dl, global_step=iter_ip1)
                # visualizer.print_current_metrics(epoch, metrics, phase='test')
                visualizer.print_current_metrics(iter_ip1, metrics, phase='test')
                # print(metrics)
                
                cprint(f'[*] End of steps %d \t Time Taken: %d sec \n%s' %
                    (
                        iter_ip1,
                        time.time() - iter_start_time,
                        os.path.abspath( os.path.join(opt.logs_dir, opt.name) )
                    ), 'blue', attrs=['bold']
                    )
        # model.update_learning_rate()

        pbar.update(1)
        
def main():
    ## set random seed
    torch.backends.cudnn.benchmark = False     
    torch.backends.cudnn.deterministic = True
    # seed = 42
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)

    # this will parse args, setup log_dirs, multi-gpus
    opt = TrainOptions().parse_and_setup()
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
    model = create_model(opt)
    cprint(f'[*] "{opt.model}" initialized.', 'cyan')

    # visualizer
    # visualizer = Visualizer(opt)
    # if get_rank() == 0:
    #     visualizer.setup_io()

    # save model and dataset files
    if get_rank() == 0:
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
        
        if opt.cvae_cfg is not None:
            cvae_cfg = opt.cvae_cfg
            cfg_out = os.path.join(expr_dir, os.path.basename(cvae_cfg))
            os.system(f'cp {cvae_cfg} {cfg_out}')

    train_main_worker(opt, model, train_dl, test_dl)

if __name__ == "__main__":
    main()


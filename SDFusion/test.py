import os
import time
import inspect

from termcolor import colored, cprint
from tqdm import tqdm

import torch.backends.cudnn as cudnn
# cudnn.benchmark = True

from options.train_options import TrainOptions
from options.test_options import TestOptions
from datasets.dataloader import CreateDataLoader, get_data_generator
from models.base_model import create_model

import torch
import random
import numpy as np

from utils.visualizer import Visualizer

cuda_avail = torch.cuda.is_available()
# import pdb; pdb.set_trace()
print(f"CUDA TORCH AVAILABLE: {cuda_avail}")


def eval_main_worker(opt, model, test_dl, visualizer: Visualizer):

	for i, test_data in tqdm(enumerate(test_dl)):

		# print(test_data['path'])

		if opt.guided_inference:
			model.guided_inference(test_data, ddim_eta=opt.ddim_eta, ddim_steps=opt.ddim_steps)
		else:
			model.inference(test_data, ddim_eta=opt.ddim_eta, ddim_steps=opt.ddim_steps)
		
		visualizer.display_current_results(model.get_current_visuals(), i, phase='test')

	model.logger.log(f'Collision loss: {model.collision_loss_meter.avg:.4f}, contact loss: {model.contact_loss_meter.avg:.4f}')

if __name__ == "__main__":
	## set random seed
	torch.backends.cudnn.benchmark = False     
	torch.backends.cudnn.deterministic = True
	# seed = 2024
	# random.seed(seed)
	# np.random.seed(seed)
	# torch.manual_seed(seed)
	# torch.cuda.manual_seed(seed)
	# torch.cuda.manual_seed_all(seed)

	# this will parse args, setup log_dirs, multi-gpus
	opt = TestOptions().parse_and_setup()

	# get current time, print at terminal. easier to track exp
	from datetime import datetime
	opt.exp_time = datetime.now().strftime('%Y-%m-%dT%H-%M')

	train_dl, test_dl = CreateDataLoader(opt)
	test_ds = test_dl.dataset

	cprint('[*] # testing images = %d' % len(test_ds), 'yellow')

	# main loop
	model = create_model(opt)
	cprint(f'[*] "{opt.model}" initialized.', 'cyan')

	# visualizer
	visualizer = Visualizer(opt)
	visualizer.setup_io()

	# save model and dataset files
	expr_dir = '%s/%s' % (opt.logs_dir, opt.name)
	model_f = inspect.getfile(model.__class__)
	dset_f = inspect.getfile(test_ds.__class__)
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

	eval_main_worker(opt, model, test_dl, visualizer)

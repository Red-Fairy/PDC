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

if __name__ == '__main__':
	opt = TestOptions().parse_and_setup()

	train_dl, test_dl = CreateDataLoader(opt)

	model = create_model(opt)

	visualizer = Visualizer(opt)
	visualizer.setup_io()

	for i, test_data in tqdm(enumerate(test_dl)):

		model.inference(test_data)

		visualizer.display_current_results(model.get_current_visuals(), i, phase='test')

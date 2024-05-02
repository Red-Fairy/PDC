import os
import argparse

from termcolor import colored
from omegaconf import OmegaConf

import torch
from torch.utils.tensorboard import SummaryWriter

from accelerate import Accelerator

class BaseOptions():
	def __init__(self):
		self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
		self.initialized = False

	def initialize(self):
		# hyper parameters
		self.parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
		self.parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

		# log stuff
		self.parser.add_argument('--logs_dir', type=str, default='./logs', help='the root of the logs dir. All training logs are saved here')
		self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')

		# dataset stuff
		self.parser.add_argument('--dataroot', type=str, default=None, help='root dir for data. if None, specify by `hostname` in configs/paths.py')
		self.parser.add_argument('--bboxroot', type=str, default=None, help='root dir for data. if None, specify by `hostname` in configs/paths.py')
		self.parser.add_argument('--dataset_mode', type=str, default='snet', help='chooses how datasets are loaded. [mnist, snet, abc, snet-abc]')
		self.parser.add_argument('--res', type=int, default=64, help='dataset resolution')
		self.parser.add_argument('--cat', type=str, default='slider_drawer', help='category for shapenet')
		self.parser.add_argument('--trunc_thres', type=float, default=0.2, help='threshold for truncated sdf.')
		
		self.parser.add_argument('--ratio', type=float, default=1., help='ratio of the dataset to use. for debugging and overfitting')
		self.parser.add_argument('--max_dataset_size', default=2147483648, type=int, help='chooses the maximum dataset size.')
		self.parser.add_argument('--nThreads', default=9, type=int, help='# threads for loading data')        
		self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')

		############## START: model related options ################
		self.parser.add_argument(
							'--model', type=str, default='sdfusion',
							# choices=['vqvae', 'sdfusion', 'sdfusion-img2shape', 'sdfusion-txt2shape','sdfusion-mm2shape','sdfusion-bbox2shape', ],
							help='chooses which model to use.'
						)

		# diffusion stuff
		self.parser.add_argument('--df_cfg', type=str, default='configs/sdfusion_snet.yaml', help="diffusion model's config file")
		self.parser.add_argument('--ddim_steps', type=int, default=10, help='steps for ddim sampler')
		self.parser.add_argument('--ddim_eta', type=float, default=0.0)
		self.parser.add_argument('--uc_scale', type=float, default=3.0, help='scale for classifier-free guidance')
		self.parser.add_argument('--uc_ply_scale', type=float, default=3.0, help='scale for ply guidance')
		self.parser.add_argument('--uc_bbox_scale', type=float, default=3.0, help='scale for bbox guidance')
		
		# vqvae stuff
		self.parser.add_argument('--vq_model', type=str, default='vqvae', help='for choosing the vqvae model to use.')
		self.parser.add_argument('--vq_cfg', type=str, default='configs/vqvae_snet.yaml', help='vqvae model config file')
		self.parser.add_argument('--vq_dset', type=str, default=None, help='dataset vqvae originally trained on')
		self.parser.add_argument('--vq_cat', type=str, default=None, help='dataset category vqvae originally trained on')
		self.parser.add_argument('--vq_ckpt', type=str, default=None, help='vqvae ckpt to load.')

		# condition model
		self.parser.add_argument('--cond_ckpt', type=str, default=None, help='condition model ckpt to load.')
		############## END: model related options ################

		# misc
		self.parser.add_argument('--debug', default='0', type=str, choices=['0', '1'], help='if true, debug mode')
		self.parser.add_argument('--seed', default=111, type=int, help='seed')

		# multi-gpu stuff
		self.parser.add_argument("--backend", type=str, default="gloo", help="which backend to use")
		self.parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")
		
		# data version
		self.parser.add_argument('--data_version', type=str, default='v.0.2', help='data version for dataset')

		# condition
		self.parser.add_argument('--bbox_cond', action='store_true', help='if true, use bbox condition')
		self.parser.add_argument('--ply_cond', action='store_true', help='if true, use pointcloud condition')
		self.parser.add_argument('--ply_bbox_cond', action='store_true', help='if true, use both pointcloud and bbox condition')
		self.parser.add_argument('--ply_rotate', action='store_true', help='if true, rotate the input pointcloud')
		self.parser.add_argument('--joint_rotate', action='store_true', help='if true, rotate the input pointcloud')
		self.parser.add_argument('--ply_norm', action='store_true', help='if true, normalize the input pointcloud')

		# continue_train or test-time load_iter
		self.parser.add_argument('--load_iter', default='latest', type=str, help='which iter to load?')

		# visual mode
		self.parser.add_argument('--visual_mode', type=str, choices=['sdf', 'mesh'], default='mesh', 
								help='sdf or mesh, if sdf move the point cloud to the center; \
									if mesh, both point cloud and mesh are translated')
		self.parser.add_argument('--visual_normalize', action='store_true', help='if true, rotate point cloud and mesh to the canonical pose')
		
		# use mobility constraint during inference/refinement
		self.parser.add_argument('--use_mobility_constraint', action='store_true', help='use mobility constraint')
		self.parser.add_argument('--mobility_sample_count', type=int, default=32, help='mobility sample count')
		self.parser.add_argument('--mobility_type', choices=['translation', 'rotation'], default='translation', 
								 help='mobility type, e.g, slider drawer is translation, hinge door is rotation')
		self.parser.add_argument('--use_predicted_scale', action='store_true', help='use predicted scale for mobility constraint')
		self.parser.add_argument('--use_predicted_volume', action='store_true', help='use predicted volume for mobility constraint')
		self.parser.add_argument('--guided_inference', action='store_true', help='use guided inference')
		self.parser.add_argument('--test_description', type=str, default=None, help='test description')
		self.parser.add_argument('--loss_collision_weight', type=float, default=1.0, help='collision loss weight')
		self.parser.add_argument('--loss_contact_weight', type=float, default=10000.0, help='contact loss weight')
		self.parser.add_argument('--haoran', action='store_true', help='haoran dataset')

		# resize factor for parts
		self.parser.add_argument('--scale_mode', choices=['volume', 'max_extent'], default='max_extent', help='scale mode for parts')

		# rotation angle of input point cloud during inference
		self.parser.add_argument('--rotate_angle', type=float, default=None, help='rotation angle of input point cloud during inference')

		# test a single model
		self.parser.add_argument('--model_id', default=None, type=str, help='model id to optimize')

		self.initialized = True

	def parse_and_setup(self, accelerator: Accelerator = None):
		import sys
		cmd = ' '.join(sys.argv)
		print(f'python {cmd}')

		if not self.initialized:
			self.initialize()

		self.opt = self.parser.parse_args()
		self.opt.isTrain = self.isTrain   # train or test

		if self.opt.isTrain:
			self.opt.phase = 'train'
		else:
			self.opt.phase = 'test'

		# make experiment dir
		expr_dir = os.path.join(self.opt.logs_dir, self.opt.name)
		if (accelerator is None or accelerator.is_main_process) and not os.path.exists(expr_dir):
			os.makedirs(expr_dir)
		
		ckpt_dir = os.path.join(self.opt.logs_dir, self.opt.name, 'ckpt')
		self.opt.ckpt_dir = ckpt_dir
		if (accelerator is None or accelerator.is_main_process) and not os.path.exists(ckpt_dir):
			os.makedirs(ckpt_dir)

		rotate_string = '' if not self.opt.ply_cond else f'_rotate{self.opt.rotate_angle}' if self.opt.rotate_angle is not None else '_rotate'
		self.opt.img_dir = os.path.join(expr_dir, 'train_visuals') if self.isTrain else \
							os.path.join(expr_dir, f'test{self.opt.testdir}_{self.opt.load_iter}{rotate_string}_scale{self.opt.uc_scale}_eta{self.opt.ddim_eta}_steps{self.opt.ddim_steps}')
		self.opt.img_dir += '_predscale' if self.opt.use_predicted_scale else ''
		self.opt.img_dir += '_extent' if self.opt.scale_mode == 'max_extent' else '_volume'
		self.opt.img_dir += '_mobility' if self.opt.use_mobility_constraint else ''
		self.opt.img_dir += '_guided' if self.opt.guided_inference else ''
		self.opt.img_dir += f'_{self.opt.model_id}' if self.opt.model_id is not None else ''
		self.opt.img_dir += f'_{self.opt.test_description}' if self.opt.test_description is not None else ''
		os.makedirs(self.opt.img_dir, exist_ok=True)

		# print args
		if (accelerator is None or accelerator.is_main_process):
			args = vars(self.opt)
			print('------------ Options -------------')
			for k, v in sorted(args.items()):
				print('%s: %s' % (str(k), str(v)))
			print('-------------- End ----------------')
			
			file_name = os.path.join(expr_dir, 'opt.txt')
			with open(file_name, 'wt') as opt_file:
				opt_file.write('------------ Options -------------\n')
				for k, v in sorted(args.items()):
					opt_file.write('%s: %s\n' % (str(k), str(v)))
				opt_file.write('-------------- End ----------------\n')

		# tensorboard writer
		tb_dir = '%s/tboard' % expr_dir
		if (accelerator is None or accelerator.is_main_process) and not os.path.exists(tb_dir):
			os.makedirs(tb_dir)

		self.opt.tb_dir = tb_dir
		writer = SummaryWriter(log_dir=tb_dir)
		self.opt.writer = writer

		return self.opt

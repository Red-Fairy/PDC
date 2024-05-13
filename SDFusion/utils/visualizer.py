import pickle
from collections import OrderedDict
import os
import ntpath
import time

from termcolor import colored
from . import util

import torch
import imageio
import numpy as np
import cv2
import matplotlib.pyplot as plt

import open3d
import h5py

def parse_line(line):
	info_d = {}

	l1, l2 = line.split(') ')
	l1 = l1.replace('(', '')
	l1 = l1.split(', ')

	l2 = l2.replace('(', '')
	l2 = l2.split(' ')

	info_d = {}
	for s in l1:
		
		k, v = s.split(': ')
		
		
		if k in ['epoch', 'iters']:
			info_d[k] = int(v)
		else:
			info_d[k] = float(v)

	l2_keys = l2[0::2]
	l2_vals = l2[1::2]
	
	for k, v in zip(l2_keys, l2_vals):
		k = k.replace(':','')
		info_d[k] = float(v)

	return info_d


class Visualizer():
	def __init__(self, opt):
		# self.opt = opt
		self.isTrain = opt.isTrain
		self.gif_fps = 4

		self.log_dir = os.path.join(opt.logs_dir, opt.name)

		self.img_dir = opt.img_dir
		
		if 'vqvae' in opt.model:
			os.makedirs(os.path.join(self.img_dir, 'meshes'), exist_ok=True)
			os.makedirs(os.path.join(self.img_dir, 'meshes_recon'), exist_ok=True)
		if 'sdfusion' in opt.model:
			os.makedirs(os.path.join(self.img_dir, 'meshes'), exist_ok=True)
			os.makedirs(os.path.join(self.img_dir, 'meshes_canonical'), exist_ok=True)
			os.makedirs(os.path.join(self.img_dir, 'pcd'), exist_ok=True)
		if 'cvae' in opt.model:
			os.makedirs(os.path.join(self.img_dir, 'meshes'), exist_ok=True)
			os.makedirs(os.path.join(self.img_dir, 'meshes_pred'), exist_ok=True)
			os.makedirs(os.path.join(self.img_dir, 'meshes_canonical'), exist_ok=True)
			os.makedirs(os.path.join(self.img_dir, 'pcd'), exist_ok=True)

		self.name = opt.name
		self.opt = opt

		self.diversity_count = 0

	def setup_io(self):

		# if self.isTrain:
		print('[*] create image directory:\n%s...' % os.path.abspath(self.img_dir) )
		util.mkdirs([self.img_dir])
		
		if self.isTrain:
			self.log_name = os.path.join(self.log_dir, 'loss_log.txt')
			# with open(self.log_name, "a") as log_file, append to the file
			with open(self.log_name, "a") as log_file:
				now = time.strftime("%c")
				log_file.write('================ Training Loss (%s) ================\n' % now)

	def reset(self):
		self.saved = False

	def print_current_errors(self, current_iters, errors, t):
		# message = '(GPU: %s, epoch: %d, iters: %d, time: %.3f) ' % (self.opt.gpu_ids_str, t)
		# message = f"[{self.opt.exp_time}] (GPU: {self.opt.gpu_ids_str}, iters: {current_iters}, time: {t:.3f}) "
		message = f"[{self.opt.name}] iters: {current_iters}, time: {t:.3f} "
		for k, v in errors.items():
			message += '%s: %.6f ' % (k, v)

		print(colored(message, 'magenta'))
		with open(self.log_name, "a") as log_file:
			log_file.write('%s\n' % message)

		self.log_tensorboard_errors(errors, current_iters)

	def print_current_metrics(self, current_iters, metrics, phase):
		# message = f'([{phase}] GPU: {}, steps: %d) ' % (phase, self.opt.gpu_ids_str, current_iters)
		# message = f'([{self.opt.exp_time}] [{phase}] GPU: {self.opt.gpu_ids_str}, steps: {current_iters}) '
		message = f'([{self.opt.name}] [{phase}] GPU: {self.opt.gpu_ids}, steps: {current_iters}) '
		for k, v in metrics.items():
			message += '%s: %.3f ' % (k, v)

		print(colored(message, 'yellow'))
		with open(self.log_name, "a") as log_file:
			log_file.write('%s\n' % message)

		# self.log_tensorboard_metrics(metrics, epoch, phase)
		self.log_tensorboard_metrics(metrics, current_iters, phase)

	def get_instance_label(self, i):
		if self.isTrain:
			return '_' + str(i)
		elif self.opt.test_diversity:
			return '_' + str(self.diversity_count + i)
		else:
			return ''

	def display_current_results(self, visuals, current_iters, im_name='', phase='train'):

		paths = visuals['paths']
		object_ids = [path.split('/')[-1].split('_')[0] for path in paths]
		part_ids = [path.split('/')[-1].split('_')[1].split('.')[0] for path in paths]

		filename_format = f'{phase}_step{current_iters:05d}' + '_{}_{}{}.{}' if self.opt.isTrain else '{}_{}{}.{}'
		
		if 'meshes' in visuals:
			visual_meshes = visuals['meshes']

			if self.opt.visual_mode == 'mesh':
				for i, mesh in enumerate(visual_meshes):
					instance_label = self.get_instance_label(i)
					if 'ply_translation' in visuals:
						part_scale = visuals['part_scale']
						part_translation = visuals['part_translation']
				
						# print(f'mesh {i} max extent: {np.max(mesh.extents)}')
						mesh.apply_scale((part_scale[i], part_scale[i], part_scale[i]))
						mesh.apply_translation(part_translation[i])
						
						# save the mesh under canonical pose
						mesh_path = os.path.join(self.img_dir, 'meshes_canonical', filename_format.format(object_ids[i], part_ids[i], instance_label, 'obj'))
						mesh.export(mesh_path, 'obj')

						# rotate the mesh by 'ply_rotation' to make them align with the point cloud
						mesh.apply_transform(visuals['ply_rotation'][i])

					mesh_path = os.path.join(self.img_dir, 'meshes', filename_format.format(object_ids[i], part_ids[i], instance_label, 'obj'))
					mesh.export(mesh_path, 'obj')
		
		if 'meshes_recon' in visuals:
			for i, visual_mesh in enumerate(visuals['meshes_recon']):
				instance_label = self.get_instance_label(i)
				mesh_path = os.path.join(self.img_dir, 'meshes_recon', filename_format.format(object_ids[i], part_ids[i], instance_label, 'obj'))
				visual_mesh.export(mesh_path, 'obj')
			
		if 'meshes_pred' in visuals:
			part_scale = visuals['part_scale'][0]
			part_translation = visuals['part_translation'][0]
			for i, visual_mesh in enumerate(visuals['meshes_pred']):
				visual_mesh.apply_scale((part_scale, part_scale, part_scale))
				visual_mesh.apply_translation(part_translation)
				mesh_path = os.path.join(self.img_dir, filename_format.format(object_ids[0], part_ids[0], i, 'pred.obj'))
				visual_mesh.export(mesh_path, 'obj')

		if 'points' in visuals:
			# save the visualized ply files, points are stored in visuals['points']
			for i in range(visuals['points'].shape[0]):
				ply_file = open3d.geometry.PointCloud()
				
				if 'ply_translation' in visuals:
					if self.opt.visual_mode == 'sdf':
						points = visuals['points'][i]
						# if 'ply_rotation' in visuals and self.opt.visual_normalize:
						# 	points = np.matmul(visuals['ply_rotation'][i][:3, :3].T, points)
						points = points + visuals['ply_translation'][i][:, None]
						points = points - visuals['part_translation'][i][:, None]
						points = points / visuals['part_scale'][i]
						ply_file.points = open3d.utility.Vector3dVector(points.T)
					elif self.opt.visual_mode == 'mesh':
						points = visuals['points'][i]
						# if 'ply_rotation' in visuals and self.opt.visual_normalize:
						# 	points = np.matmul(visuals['ply_rotation'][i][:3, :3].T, points)
						points = points + visuals['ply_translation'][i][:, None]
						ply_file.points = open3d.utility.Vector3dVector(points.T)
				
				# if not self.opt.test_diversity or (self.diversity_count == 0 and i == 0):
				instance_label = self.get_instance_label(i)
				ply_path = os.path.join(self.img_dir, 'pcd', filename_format.format(object_ids[i], part_ids[i], instance_label, 'ply'))
				open3d.io.write_point_cloud(ply_path, ply_file)

		if self.opt.visual_mode == 'sdf': # save the sdf file
			for i in range(visuals['sdf'].shape[0]):
				instance_label = self.get_instance_label(i)
				sdf_path = os.path.join(self.img_dir, filename_format.format(object_ids[0], part_ids[0], instance_label, 'sdf'))
				# save as h5py file
				with h5py.File(sdf_path, 'w') as f:
					f.create_dataset('sdf', data=visuals['sdf'][i], compression='gzip', compression_opts=4)

		# if self.opt.bbox_cond:
		# 	bboxes = visuals['bboxes']
		# 	data_dict = {
		# 		'bboxes': bboxes,
		# 	}
		# 	np.save(f'{self.img_dir}/{phase}_step{current_iters:05d}_meta.npy', data_dict, allow_pickle=True)
			
		# write images to disk
		if 'img' in visuals:
			for label, image_numpy in visuals['img'].items():
				suffix = f'{phase}_step{current_iters:05d}_{label}.png'
				img_path = os.path.join(self.img_dir, suffix)
				util.save_image(image_numpy, img_path)
			# log to tensorboard
			self.log_tensorboard_visuals(visuals, current_iters, phase=phase)

		if not self.opt.isTrain and self.opt.test_diversity:
			self.diversity_count += len(visuals['meshes'])
			if self.diversity_count >= self.opt.diversity_count:
				self.diversity_count = 0

	def log_tensorboard_visuals(self, visuals, cur_step, labels_while_list=None, phase='train'):
		writer = self.opt.writer

		if labels_while_list is None:
			labels_while_list = []

		# NOTE: we have ('text', text_data) as visuals now
		visual_img = visuals['img']
		for ix, (label, image_numpy) in enumerate(visual_img.items()):
			if image_numpy.shape[2] == 4:
				image_numpy = image_numpy[:, :, :3]

			if label not in labels_while_list:
				# writer.add_image('vis/%d-%s' % (ix+1, label), image_numpy, global_step=cur_step, dataformats='HWC')
				writer.add_image('%s/%d-%s' % (phase, ix+1, label), image_numpy, global_step=cur_step, dataformats='HWC')
			else:
				pass
				# log the unwanted image just in case
				# writer.add_image('other/%s' % (label), image_numpy, global_step=cur_step, dataformats='HWC')

	def log_tensorboard_errors(self, errors, cur_step):
		writer = self.opt.writer

		for label, error in errors.items():
			writer.add_scalar('losses/%s' % label, error, cur_step)

	def log_tensorboard_metrics(self, metrics, cur_step, phase):
		writer = self.opt.writer

		for label, value in metrics.items():
			writer.add_scalar('metrics/%s-%s' % (phase, label), value, cur_step)
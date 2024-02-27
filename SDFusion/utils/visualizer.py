
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

        if self.isTrain:
            # self.log_dir = os.path.join(opt.checkpoints_dir, opt.name)
            self.log_dir = os.path.join(opt.logs_dir, opt.name)
        else:
            self.log_dir = os.path.join(opt.results_dir, opt.name)

        self.img_dir = os.path.join(self.log_dir, 'images')
        self.name = opt.name
        self.opt = opt

    def setup_io(self):
        
        print('[*] create image directory:\n%s...' % os.path.abspath(self.img_dir) )
        util.mkdirs([self.img_dir])
        # self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')

        if self.isTrain:
            self.log_name = os.path.join(self.log_dir, 'loss_log.txt')
            # with open(self.log_name, "a") as log_file:
            with open(self.log_name, "w") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        self.saved = False

    def print_current_errors(self, current_iters, errors, t):
        # message = '(GPU: %s, epoch: %d, iters: %d, time: %.3f) ' % (self.opt.gpu_ids_str, t)
        # message = f"[{self.opt.exp_time}] (GPU: {self.opt.gpu_ids_str}, iters: {current_iters}, time: {t:.3f}) "
        message = f"[{self.opt.name}] (GPU: {self.opt.gpu_ids_str}, iters: {current_iters}, time: {t:.3f}) "
        for k, v in errors.items():
            message += '%s: %.6f ' % (k, v)

        print(colored(message, 'magenta'))
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

        self.log_tensorboard_errors(errors, current_iters)

    def print_current_metrics(self, current_iters, metrics, phase):
        # message = f'([{phase}] GPU: {}, steps: %d) ' % (phase, self.opt.gpu_ids_str, current_iters)
        # message = f'([{self.opt.exp_time}] [{phase}] GPU: {self.opt.gpu_ids_str}, steps: {current_iters}) '
        message = f'([{self.opt.name}] [{phase}] GPU: {self.opt.gpu_ids_str}, steps: {current_iters}) '
        for k, v in metrics.items():
            message += '%s: %.3f ' % (k, v)

        print(colored(message, 'yellow'))
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

        # self.log_tensorboard_metrics(metrics, epoch, phase)
        self.log_tensorboard_metrics(metrics, current_iters, phase)

    def display_current_results(self, visuals, current_iters, im_name='', phase='train'):
        visual_img = visuals['img']
        visual_meshes = visuals['meshes']
        # paths = visuals['paths']
        if self.opt.ply_cond:
            ply_paths = [path.replace('part_sdf', 'part_ply').replace('.h5', '.ply') for path in visuals['paths']]
            # create a symbolic link
            for i, path in enumerate(ply_paths):
                dst_path = os.path.join(self.img_dir, f'{phase}_step{current_iters:05d}_{im_name}_{i}.ply')
                if not os.path.exists(dst_path):
                    os.system(f'ln -s {path} {dst_path}')
        if self.opt.bbox_cond:
            bboxes = visuals['bboxes']
            data_dict = {
                'bboxes': bboxes,
            }
            np.save(f'{self.img_dir}/{phase}_step{current_iters:05d}_{im_name}_meta.npy', data_dict, allow_pickle=True)
        # write images to disk
        for label, image_numpy in visual_img.items():
            img_path = os.path.join(self.img_dir, f'{phase}_step{current_iters:05d}_{label}_{im_name}.png')
            util.save_image(image_numpy, img_path)
        for mesh_i, visual_mesh in enumerate(visual_meshes):
            mesh_path = os.path.join(self.img_dir, f'{phase}_step{current_iters:05d}_mesh_{im_name}_{mesh_i}.obj')
            visual_mesh.export(mesh_path, 'obj')
        # log to tensorboard
        self.log_tensorboard_visuals(visuals, current_iters, phase=phase)

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
import os
from termcolor import colored, cprint
import torch
import utils.util as util

def create_model(opt, accelerator=None, input_instance=None):
    model = None

    if opt.model == 'vqvae':
        if accelerator is None:
            from models.vqvae_model import VQVAEModel
            model = VQVAEModel(opt)
        else:
            from models.vqvae_acc_model import VQVAEAccModel
            model = VQVAEAccModel(opt, accelerator)
    
    elif opt.model == 'sdfusion':
        if accelerator is None:
            from models.sdfusion_model import SDFusionModel
            model = SDFusionModel(opt)
        else:
            from models.sdfusion_acc_model import SDFusionModelAcc
            model = SDFusionModelAcc(opt, accelerator)

    elif opt.model == 'sdfusion-ply2shape':
        if accelerator is None:
            from models.sdfusion_ply2shape_model import SDFusionModelPly2Shape
            model = SDFusionModelPly2Shape(opt)
        else:
            from models.sdfusion_ply2shape_acc_model import SDFusionModelPly2ShapeAcc
            model = SDFusionModelPly2ShapeAcc(opt, accelerator)

    elif opt.model == 'sdfusion-plybbox2shape':
        if accelerator is None:
            from models.sdfusion_plybbox2shape_model import SDFusionModelPlyBBox2Shape
            model = SDFusionModelPlyBBox2Shape(opt)
        else:
            from models.sdfusion_plybbox2shape_acc_model import SDFusionModelPlyBBox2ShapeAcc
            model = SDFusionModelPlyBBox2ShapeAcc(opt, accelerator)

    elif opt.model == 'sdfusion-ply2shape-refine':
        if accelerator is None:
            raise ValueError("Refine model must be used with accelerator")
        else:
            from models.sdfusion_ply2shape_refine_acc_model import SDFusionModelPly2ShapeRefineAcc
            model = SDFusionModelPly2ShapeRefineAcc(opt, accelerator, input_instance)
        
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)

    cprint("[*] Model has been created: %s" % model.name(), 'blue')
    return model


# modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
class BaseModel():
    def name(self):
        return 'BaseModel'

    def __init__(self, opt):
        self.opt = opt
        # self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        # self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor

        self.model_names = []
        self.epoch_labels = []
        self.optimizers = []

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    # define the optimizers
    def set_optimizers(self):
        pass

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        # print('[*] learning rate = %.7f' % lr)

    def eval(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def train(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()

    # print network information
    def print_networks(self, verbose=False):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def tnsrs2ims(self, tensor_names):
        ims = []
        for name in tensor_names:
            if isinstance(name, str):
                var = getattr(self, name)
                ims.append(util.tensor2im(var.data))
        return ims

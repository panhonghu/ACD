import os
import torch
import random
import torch.nn as nn
import util
from .patch_gan import *
from itertools import chain
from .diffusion import Diffusion
from torch.autograd import Variable


class ConditionDiffusion(nn.Module):
    def __init__(self, args):
        super(ConditionDiffusion, self).__init__()
        self.args = args
        self.device = 'cuda' if len(args.gpu) > 0 else 'cpu'
        self.gpu = [0] if len(args.gpu)==1 else [int(i) for i in args.gpu.split(',')]
        self.is_training = args.is_training
        # Set up RealNVP generators
        self.generator = Diffusion(args, self.device)
        if self.is_training:
            # Set up discriminators
            self.d_rgb = PatchGAN(args, return_binary=True)
            self.d_ir = PatchGAN(args, return_binary=True)
            self._data_parallel()
            # Set up loss functions
            self.max_grad_norm = 0.0
            self.modality_loss_fn = torch.nn.MSELoss()
            self.modality_loss_fn = self.modality_loss_fn.to(self.device)
            Tensor = torch.cuda.FloatTensor if self.device=='cuda' else torch.Tensor
            self.target_real = Variable(Tensor(args.batch_size).fill_(1.0), requires_grad=False)
            self.target_fake = Variable(Tensor(args.batch_size).fill_(0.0), requires_grad=False)
            self.opt_g = torch.optim.Adam(self.generator.model.parameters(), lr=2e-4)
            self.opt_d = torch.optim.Adam(chain(self.d_ir.parameters(), self.d_rgb.parameters()), lr=2e-4)
        else:
            self._data_parallel()

        # rgb modality
        self.rgb = None
        self.rgb_HF = None
        self.rgb_fake = None
        # ir modality
        self.ir = None
        self.ir_HF = None
        self.ir_target = None
        self.ir_fake = None
        # Generator losses
        self.loss_diffusion = None
        self.loss_diffusion_ir = None
        self.loss_diffusion_rgb = None
        self.loss_g_mod_rgb = None
        self.loss_g_mod_ir = None
        self.loss_g_mod = None
        # Discriminator losses
        self.loss_d_mod_rgb = None
        self.loss_d_mod_ir = None
        self.loss_d = None

    def set_inputs(self, rgb_input, rgb_HF_input, ir_input, ir_HF_input, rgb_target, ir_target):
        """Set the inputs prior to a forward pass through the network.
        Args:
            rgb_input: Tensor with rgb input
            ir_input: Tensor with ir input
        """
        self.rgb = rgb_input.to(self.device)
        self.rgb_HF = rgb_HF_input.to(self.device)
        self.ir = ir_input.to(self.device)
        self.ir_HF = ir_HF_input.to(self.device)
        self.rgb_target = rgb_target.to(self.device)
        self.ir_target = ir_target.to(self.device)

    def _data_parallel(self):
        self.generator.model = nn.DataParallel(self.generator.model, device_ids=self.gpu).to(self.device)
        self.d_rgb = nn.DataParallel(self.d_rgb, device_ids=self.gpu).to(self.device)
        self.d_ir = nn.DataParallel(self.d_ir, device_ids=self.gpu).to(self.device)

    def __backward_D(self):
        self.generator.model.eval()
        self.d_rgb.train()
        self.d_ir.train()
        self.loss_d_mod_rgb = self.modality_loss_fn(self.d_rgb(self.rgb_fake.detach()), self.target_fake) + \
                              self.modality_loss_fn(self.d_rgb(self.rgb), self.target_real)
        self.loss_d_mod_ir = self.modality_loss_fn(self.d_ir(self.ir_fake.detach()), self.target_fake) + \
                              self.modality_loss_fn(self.d_ir(self.ir), self.target_real)
        self.loss_d = self.loss_d_mod_rgb + self.loss_d_mod_ir
        self.loss_d.backward()

    def __backward_G(self):
        self.generator.model.train()
        self.d_rgb.eval()
        self.d_ir.eval()
        self.loss_diffusion_rgb = self.generator.train(x=self.rgb, con_x=self.rgb_HF, modality="rgb")
        self.loss_diffusion_ir = self.generator.train(x=self.ir, con_x=self.ir_HF, modality="ir")
        self.loss_diffusion = self.loss_diffusion_rgb + self.loss_diffusion_ir
        if random.random()>0.5:
            HF = self.rgb_HF
        else:
            HF = self.ir_HF
        self.rgb_fake = self.generator.sample(x0=self.rgb, con_x=HF, modality="rgb")
        self.loss_g_mod_rgb = self.modality_loss_fn(self.d_rgb(self.rgb_fake), self.target_real)
        self.ir_fake = self.generator.sample(x0=self.ir, con_x=HF, modality="ir")
        self.loss_g_mod_ir = self.modality_loss_fn(self.d_ir(self.ir_fake), self.target_real)
        self.loss_g_mod = self.loss_g_mod_rgb + self.loss_g_mod_ir
        self.loss_g = 0.01*self.loss_diffusion + self.loss_g_mod
        self.loss_g.backward()

    def __backward_diffusion(self):
        self.generator.model.train()
        self.d_rgb.eval()
        self.d_ir.eval()
        self.loss_diffusion_rgb = self.generator.train(x=self.rgb, con_x=self.rgb_HF, modality="rgb")
        self.loss_diffusion_ir = self.generator.train(x=self.ir, con_x=self.ir_HF, modality="ir")
        self.loss_diffusion = self.loss_diffusion_rgb + self.loss_diffusion_ir
        self.loss_diffusion.backward()
        print(self.loss_diffusion)

    def train_iter(self, epoch):
        if epoch <= self.args.diffusion_only_epoch:
            self.opt_g.zero_grad()
            self.__backward_diffusion()
            util.clip_grad_norm(self.opt_g, self.max_grad_norm)
            self.opt_g.step()
            return self.loss_diffusion.detach().cpu(), 0
        else:
            ## train generator
            for i in range(self.args.g_steps):
                self.opt_g.zero_grad()
                self.__backward_G()
                util.clip_grad_norm(self.opt_g, self.max_grad_norm)
                self.opt_g.step()
            ## train discriminators
            for i in range(self.args.d_steps):
                self.opt_d.zero_grad()
                self.__backward_D()
                util.clip_grad_norm(self.opt_d, self.max_grad_norm)
                self.opt_d.step()
            return self.loss_g.detach().cpu(), self.loss_d.detach().cpu()


    def __backward_D_middle(self):
        self.generator.model.eval()
        self.d_rgb.train()
        self.d_ir.train()
        self.loss_d_mod_rgb = self.modality_loss_fn(self.d_rgb(self.rgb2mid.detach()), self.target_fake) + \
                              self.modality_loss_fn(self.d_rgb(self.rgb), self.target_real)
        self.loss_d_mod_ir = self.modality_loss_fn(self.d_ir(self.ir2mid.detach()), self.target_fake) + \
                              self.modality_loss_fn(self.d_ir(self.ir), self.target_real)
        self.loss_distance = torch.sqrt((1.0*self.loss_d_mod_rgb - 2.0*self.loss_d_mod_ir)**2)
        self.loss_d = self.loss_d_mod_rgb + self.loss_d_mod_ir + 5*self.loss_distance
        # print('self.loss_d_mod_rgb -->> ', self.loss_d_mod_rgb)
        # print('self.loss_d_mod_ir -->> ', self.loss_d_mod_ir)
        # print('self.loss_distance -->> ', self.loss_distance)
        # print()
        self.loss_d.backward()

    def __backward_G_middle(self):
        self.generator.model.train()
        self.d_rgb.eval()
        self.d_ir.eval()
        self.loss_diffusion_rgb = self.generator.train(x=self.rgb, con_x=self.rgb_HF, modality="rgb")
        self.loss_diffusion_ir = self.generator.train(x=self.ir, con_x=self.ir_HF, modality="ir")
        self.loss_diffusion = self.loss_diffusion_rgb + self.loss_diffusion_ir

        self.rgb2mid = self.generator.sample(x0=self.rgb, con_x=self.rgb_HF, modality="mid")
        self.ir2mid = self.generator.sample(x0=self.ir, con_x=self.ir_HF, modality="mid")
        self.loss_g_mod_rgb = self.modality_loss_fn(self.d_rgb(self.rgb2mid), self.target_real)
        self.loss_g_mod_ir = self.modality_loss_fn(self.d_ir(self.ir2mid), self.target_real)
        self.loss_g_mod = self.loss_g_mod_rgb + self.loss_g_mod_ir

        rgb_contours = contour_extraction(self.rgb)
        rgb2mid_contours = contour_extraction(self.rgb2mid)
        self.loss_g_mid_differ_rgb = torch.sum(torch.sqrt((rgb_contours - rgb2mid_contours)**2))
        ir_contours = contour_extraction(self.ir)
        ir2mid_contours = contour_extraction(self.ir2mid)
        self.loss_g_mid_differ_ir = torch.sum(torch.sqrt((ir_contours - ir2mid_contours)**2))
        self.loss_g_mid_differ = self.loss_g_mid_differ_rgb +self.loss_g_mid_differ_ir
        self.loss_g = 0.01*self.loss_diffusion + self.loss_g_mod + 0.001*self.loss_g_mid_differ
        self.loss_g.backward()

    def train_iter_middle(self, epoch):
        ## train generator
        for i in range(self.args.g_steps):
            self.opt_g.zero_grad()
            self.__backward_G_middle()
            util.clip_grad_norm(self.opt_g, self.max_grad_norm)
            self.opt_g.step()
        ## train discriminators
        for i in range(self.args.d_steps):
            self.opt_d.zero_grad()
            self.__backward_D_middle()
            util.clip_grad_norm(self.opt_d, self.max_grad_norm)
            self.opt_d.step()
        return self.loss_g.detach().cpu(), self.loss_d.detach().cpu()


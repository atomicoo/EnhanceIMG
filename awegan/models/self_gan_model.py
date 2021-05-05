import itertools

import torch
import torch.nn as nn
import torch.optim as optim

from .base_model import BaseModel
from . import networks

import numpy as np
import cv2


class SelfGANModel(BaseModel):
    """The SelfGANModel model."""
    @staticmethod
    def modify_cmd_options(parser, is_train=True):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='degraded')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_cyc', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_L1', type=float, default=10.0, help='weight for GL1 loss (the constraint for B)')

        return parser

    def __init__(self, opt):
        """Initialize the SelfGANModel model class.

        Parameters:
            opt (Option class)-- stores all the experiment flags
        """
        BaseModel.__init__(self, opt)
        # specify the training losses
        self.loss_names = ['G_GAN', 'G_cycle', 'G_GL1', 'D_real', 'D_fake']
        # specify the images to save/display
        self.visual_names = ['real_A', 'fake_B', 'rec_A', 'constraint']
        # specify the models
        if self.is_train:
            self.model_names = ['G_A2B', 'G_B2A', 'D']
        else:
            self.model_names = ['G_A2B', 'G_B2A']

        # define networks (both Generators and discriminators)
        # define generators
        self.netG_A2B = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                          not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B2A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                          not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        # define discriminators
        if self.is_train:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.is_train:
            # define loss functions
            self.criterionGL1 = nn.L1Loss()
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = optim.Adam(itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        self.real_A = input['A'].to(self.device)
        self.constraint = input['B'].to(self.device)
        self.image_paths = input['A_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A2B(self.real_A)   # G_A2B(A)
        self.rec_A = self.netG_B2A(self.fake_B)    # G_B2A(G_A2B(A))

    def backward_D(self):
        """Calculate the loss for discriminator D"""
        # Real
        pred_real = self.netD(self.real_A)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = self.netD(self.rec_A.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        self.loss_D = (self.loss_D_real + self.loss_D_fake) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate the loss for generators G_A2B and G_B2A"""
        lambda_Cyc = self.opt.lambda_cyc
        lambda_L1 = self.opt.lambda_L1

        # G_A2B and G_B2A loss
        self.loss_G_GL1 = lambda_L1 * self.criterionGL1(self.fake_B, self.constraint)
        # GAN loss D(G_B2A(G_A2B(A)))
        self.loss_G_GAN = self.criterionGAN(self.netD(self.rec_A), True)
        # Forward cycle loss || G_B2A(G_A2B(A)) - A||
        self.loss_G_cycle = self.criterionCycle(self.rec_A, self.real_A) * lambda_Cyc
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_GL1 + self.loss_G_cycle
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A2B and G_B2A
        self.set_requires_grad([self.netD], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A2B and G_B2A's gradients to zero
        self.backward_G()             # calculate gradients for G_A2B and G_B2A
        self.optimizer_G.step()       # update G_A2B and G_B2A's weights
        # D_A and D
        self.set_requires_grad([self.netD], True)
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()        # calculate gradients for D
        self.optimizer_D.step()  # update D's weights


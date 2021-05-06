import os
import os.path as osp
import functools

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(network, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        network (network)  -- network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    network.apply(init_func)  # apply the initialization function <init_func>


def init_network(network, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        network (network)  -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        network.to(gpu_ids[0])
        network = nn.DataParallel(network, gpu_ids)  # multi-GPUs
    init_weights(network, init_type, init_gain=init_gain)
    return network


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    network = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9bs':
        network = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6bs':
        network = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        network = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        network = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_network(network, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    network = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        network = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        network = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        network = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_network(network, init_type, init_gain, gpu_ids)


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.start_epoch - opt.lr_decay_iters) / (opt.n_epochs + opt.start_epoch - opt.lr_decay_iters)
            return lr_l
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'exp':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.lr_decay_iters, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


##############################################################################
# GANLoss Classes
##############################################################################


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


##############################################################################
# Network Classes
##############################################################################


class TemporalPad(nn.Module):
    """Pad input tensor to temporal size with given padding type"""
    def __init__(self, padding_type):
        if padding_type == 'reflect':
            self.padding_type = nn.ReflectionPad2d
            self.padding_size = 1
        elif padding_type == 'replicate':
            self.padding_type = nn.ReplicationPad2d
            self.padding_size = 1
        elif padding_type == 'zero':
            self.padding_type = nn.ZeroPad2d
            self.padding_size = 0
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

    def __call__(self):
        return self.padding_type(self.padding_size)


class ResnetGenerator(nn.Module):
    """Defines a Resnet-based generator that consists of Resnet blocks 
       between a few downsampling/upsampling operations. 
       Get idea from Justin Johnson's neural style transfer project
       (https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0), "the num of resnet block `n_blocks` should not be negative"
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [  nn.ReflectionPad2d(3),
                   nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                   norm_layer(ngf),
                   nn.ReLU(True)  ]

        n_downsampling = n_upsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True)  ]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [
                ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
            ]

        for i in range(n_upsampling):  # add upsampling layers
            mult = 2 ** (n_upsampling - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                   kernel_size=3, stride=2,
                                   padding=1, output_padding=1,
                                   bias=use_bias),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True)
            ]

        model += [  nn.ReflectionPad2d(3)  ]
        model += [  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)  ]
        model += [  nn.Tanh()  ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        padw = 1 if padding_type == 'zero' else 0

        conv_block += [  TemporalPad(padding_type)()  ]  # padding
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=padw, bias=use_bias), 
            norm_layer(dim), 
            nn.ReLU(True)  ]

        if use_dropout:
            conv_block += [  nn.Dropout(0.5)  ]

        conv_block += [  TemporalPad(padding_type)()  ]  # padding
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=padw, bias=use_bias), 
            norm_layer(dim)  ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        return x + self.conv_block(x)  # add skip connections


class UnetGenerator(nn.Module):
    """Define a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        It is a recursive process to construct the U-Net from the innermost layer to the outermost layer.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        -- X -------------------- identity -------------------- Y --
           |                                                    |
           |--- downsampling -- | submodule | -- upsampling --- |
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [ downconv ]
            up = [ uprelu, upconv, nn.Tanh() ]
            model = down + [ submodule ] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [ downrelu, downconv ]
            up = [ uprelu, upconv, upnorm ]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [ downrelu, downconv, downnorm ]
            up = [ uprelu, upconv, upnorm ]

            if use_dropout:
                model = down + [ submodule ] + up + [ nn.Dropout(0.5) ]
            else:
                model = down + [ submodule ] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        # use_bias: no need to use bias as BatchNorm2d has affine parameters
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4    # kernel size
        padw = 1  # padding
        model = [  nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), 
                   nn.LeakyReLU(0.2, True)  ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            model += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        model += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        model += [  nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)  ]  # output 1 channel prediction map
        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        # use_bias: no need to use bias as BatchNorm2d has affine parameters
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)
        ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


##############################################################################
# Special Loss Classes
##############################################################################

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
# from torchvision import models
from torchvision.models.vgg import VGG


class VGGNet(VGG):
    """Define VGG Networks: vgg11, vgg13, vgg16, vgg19.

    References:
        <torchvision.models.vgg.VGG>
        https://github.com/minar09/VGG16-PyTorch/blob/master/vgg.py
        https://github.com/GitHberChen/FCN-Pytorch/blob/master/src/model/fcn.py
    """

    model_urls = {
        'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
        'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
        'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
        'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
        'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
        'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
        'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
        'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
    }

    cfg = {
        'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }

    @staticmethod
    def make_layers(cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(True)]
                else:
                    layers += [conv2d, nn.ReLU(True)]
                in_channels = v
        return nn.Sequential(*layers)

    def __init__(self, pretrained=True, model='vgg16', requires_grad=False,
                 remove_fc=True, show_params=False):
        super(VGGNet, self).__init__(VGGNet.make_layers(VGGNet.cfg[model]))

        if pretrained:
            self.load_state_dict(model_zoo.load_url(VGGNet.model_urls[model]))
        else:
            self._initialize_weights()

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        # delete redundant fully-connected layer.
        if remove_fc:
            del self.classifier

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.features(x)


def load_vggnet(model='vgg16', gpu_ids=[], **kwargs):
    """Load pretrained VGGNet model."""
    vgg = VGGNet(model=model, requires_grad=False, **kwargs)

    if len(gpu_ids) and torch.cuda.is_available():
        vgg.to(gpu_ids[0])
        vgg = nn.DataParallel(vgg, gpu_ids)  # multi-GPUs

    return vgg


def vgg_preprocess(batch, opt):
    tensortype = type(batch.data)
    (r, g, b) = torch.chunk(batch, 3, dim = 1)
    batch = torch.cat((b, g, r), dim = 1)  # convert RGB to BGR
    batch = (batch + 1) * 255 * 0.5  # [-1, 1] -> [0, 255]
    if opt.vgg_mean:
        mean = tensortype(batch.data.size())
        mean[:, 0, :, :] = 103.939
        mean[:, 1, :, :] = 116.779
        mean[:, 2, :, :] = 123.680
        batch = batch.sub(Variable(mean))  # subtract mean
    return batch


class PerceptualLoss(nn.Module):
    def __init__(self, opt, vgg):
        super(PerceptualLoss, self).__init__()
        self.opt = opt
        self.vgg = vgg
        self._set_forward_hooks_for_output(opt.vgg_choose)
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)

    def compute_vgg_loss(self, source, target):
        source_vgg = vgg_preprocess(source, self.opt)
        target_vgg = vgg_preprocess(target, self.opt)

        source_fts = self.vgg(source_vgg)
        source_fts = self.vgg.chosen_features
        target_fts = self.vgg(target_vgg)
        target_fts = self.vgg.chosen_features

        if not self.opt.no_vgg_instance:
            source_fts = self.instancenorm(source_fts)
            target_fts = self.instancenorm(target_fts)

        return torch.mean((source_fts - target_fts) ** 2)

    def _get_forward_hook_for_output(self):
        def _intermediate_output_hook(module, input, output):
            self.vgg.chosen_features = output
        return _intermediate_output_hook

    def _set_forward_hooks_for_output(self, choose='relu5_3'):
        vgg_fts_idx = 1
        for (_, m) in self.vgg.named_modules():
            if (vgg_fts_idx > 3) and isinstance(m, nn.ReLU):
                vgg_fts_idx_idx += 1
                name = "relu%d_%d" % (vgg_fts_idx, vgg_fts_idx_idx)
                if name == choose:
                    m.register_forward_hook(self._get_forward_hook_for_output())
            if isinstance(m, nn.MaxPool2d):
                vgg_fts_idx += 1
                vgg_fts_idx_idx = 0


class FCNNet(nn.Module):
    """There are some difference from original fcn

    References:
        https://github.com/GitHberChen/FCN-Pytorch/blob/master/src/model/fcn.py
    """

    map_features = {
        'vgg16': [  'conv1_1', 'conv1_1_bn', 'conv1_1_bn', 'relu1_1', 'conv1_2', 'conv1_2_bn', 'relu1_2', 'pool1',
                    'conv2_1', 'conv2_1_bn', 'conv2_1_bn', 'relu2_1', 'conv2_2', 'conv2_2_bn', 'relu2_2', 'pool2',
                    'conv3_1', 'conv3_1_bn', 'relu3_1', 'conv3_2', 'conv3_2_bn', 'relu3_2', 'conv3_3', 'conv3_3_bn', 'relu3_3', 'pool3',
                    'conv4_1', 'conv4_1_bn', 'relu4_1', 'conv4_2', 'conv4_2_bn', 'relu4_2', 'conv4_3', 'conv4_3_bn', 'relu4_3', 'pool4',
                    'conv5_1', 'conv5_1_bn', 'relu5_1', 'conv5_2', 'conv5_2_bn', 'relu5_2', 'conv5_3', 'conv5_3_bn', 'relu5_3', 'pool5'  ],
    }

    def __init__(self, n_class=21, pretrained=True, model='fcn32s',
                 backbone='vgg16', pretrained_base=True,
                 requires_grad=False, show_params=False):
        super(FCNNet, self).__init__()

        try:
            self.backbone = backbone
            pretrained_base = not pretrained and pretrained_base
            vggnet = VGGNet(pretrained=pretrained_base, model=backbone, remove_fc=True)
        except KeyError:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        self.copy_weights_from_vgg(vggnet)

        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 =  nn.ReLU(True)
        self.drop6 = nn.Dropout2d()

        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.upscore = nn.ConvTranspose2d(n_class, n_class, 64, stride=32, bias=False)

        if pretrained:
            self.load_state_dict(torch.load("./fcn32s_from_caffe.pth"))
        else:
            self._initialize_weights()

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    @staticmethod
    def get_upsampling_weight(in_channels, out_channels, kernel_size):
        """Make a 2D bilinear kernel suitable for upsampling

        References:
            https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
        """
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:kernel_size, :kernel_size]
        filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
        weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64)
        weight[range(in_channels), range(out_channels), :, :] = filt

        return torch.from_numpy(weight).float()

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if int(name[4]) <= 5:
                print(name)
                continue
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = FCNNet.get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def copy_weights_from_vgg(self, vggnet):
        features = FCNNet.map_features[self.backbone]
        if not self.backbone.endswith('_bn'):
            features = [ft for ft in features if not ft.endswith('_bn')]
        vggnet_features = list(vggnet.features)
        assert len(features) == len(vggnet_features)
        for l1, l2 in zip(features, vggnet_features):
            if l1.startswith('conv'):
                if not l1.endswith('_bn'):
                    assert isinstance(l2, nn.Conv2d)
                else:
                    assert isinstance(l2, nn.BatchNorm2d)
            if l1.startswith('relu'):
                assert isinstance(l2, nn.ReLU)
            if l1.startswith('pool'):
                assert isinstance(l2, nn.MaxPool2d)
            setattr(self, l1, l2)

    def forward(self, x):
        h = self.backbone(x)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)
        h = self.upscore(h)
        h = h[:, :, 19:19 + x.size(2), 19:19 + x.size(3)].contiguous()

        return h


def load_fcnnet(model='fcn32s', gpu_ids=[], **kwargs):
    """Load pretrained FCNNet model."""
    fcn = FCNNet(model='fcn32s', requires_grad=False, **kwargs)

    if len(gpu_ids) and torch.cuda.is_available():
        fcn.to(gpu_ids[0])
        fcn = nn.DataParallel(fcn, gpu_ids)  # multi-GPUs

    return fcn


class SemanticLoss(nn.Module):
    def __init__(self, opt, fcn):
        super(SemanticLoss, self).__init__()
        self.opt = opt
        self.fcn = fcn
        self.instancenorm = nn.InstanceNorm2d(21, affine=False)

    def compute_fcn_loss(self, source, target):
        source_fcn = vgg_preprocess(source, self.opt)
        target_fcn = vgg_preprocess(target, self.opt)

        source_fts = self.fcn(source_fcn)
        target_fts = self.fcn(target_fcn)

        source_fts = self.instancenorm(source_fts)
        target_fts = self.instancenorm(target_fts)

        return torch.mean((source_fts - target_fts) ** 2)


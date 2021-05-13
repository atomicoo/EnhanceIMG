import math
import functools
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet


##############################################################################
# Base Module Classes
##############################################################################


class TemporalPad2d():
    """Pad input tensor to temporal size with given padding type.
       Either string defining an pad function or module (e.g. nn.ZeroPad2d)
    """
    def __init__(self, padding_type='zero', padding_size=0):
        if isinstance(padding_type, str):
            if padding_type == 'reflect':
                padding_type = nn.ReflectionPad2d
            elif padding_type == 'replicate':
                padding_type = nn.ReplicationPad2d
            elif padding_type == 'zero':
                padding_type = nn.ZeroPad2d
            else:
                raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        self.padding_type = padding_type
        self.padding_size = padding_size

    def __call__(self):
        return self.padding_type(self.padding_size)


class Swish(nn.Module):
    """Swish activation: x * sigmoid(x)
       Ref: https://arxiv.org/abs/1710.05941
       The type was so huge that I could not help but try it
    """
    def __init__(self):
        super(Swish, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


class NonLinear():
    """Either string defining an activation function or module (e.g. nn.ReLU)"""
    def __init__(self, activation_type='LeakyReLU'):
        if isinstance(activation_type, str):
            if activation_type == 'leakyrelu':
                activation_type = nn.LeakyReLU
            elif activation_type == 'sigmoid':
                activation_type = nn.Sigmoid
            elif activation_type == 'swish':
                activation_type = Swish
            elif activation_type == 'relu':
                activation_type = nn.ReLU
            elif activation_type == 'elu':
                activation_type = nn.ELU
            elif activation_type == 'none':
                activation_type = nn.Identity
            else:
                raise NotImplementedError('nonlinear [%s] is not implemented' % activation_type)

        if activation_type is None:
            activation_type = nn.LeakyReLU

        if activation_type == nn.LeakyReLU:
            activation_type = functools.partial(nn.LeakyReLU, negative_slope=0.2, inplace=False)

        self.activation_type = activation_type

    def __call__(self, *args, **kwargs):
        return self.activation_type(*args, **kwargs)


class Normalize():
    """Either string defining an normalization function or module (e.g. nn.BatchNorm2d)"""
    def __init__(self, normalize_type='Batch'):

        if isinstance(normalize_type, str):
            if normalize_type == 'batch':
                normalize_type = nn.BatchNorm2d
            elif normalize_type == 'instance':
                normalize_type = nn.InstanceNorm2d
            elif normalize_type == 'none':
                normalize_type = nn.Identity
            else:
                raise NotImplementedError('normalize [%s] is not implemented' % normalize_type)

        if normalize_type is None:
            normalize_type = nn.BatchNorm2d

        if normalize_type == nn.BatchNorm2d:
            normalize_type = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        elif normalize_type == nn.InstanceNorm2d:
            normalize_type = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)

        self.normalize_type = normalize_type

    def __call__(self, *args, **kwargs):
        return self.normalize_type(*args, **kwargs)


def get_kernel(factor, kernel_type, phase, kernel_width, support=None, sigma=None):
    assert kernel_type in ['lanczos', 'gauss', 'box']

    # factor = float(factor)
    if phase == 0.5 and kernel_type != 'box': 
        kernel = np.zeros([kernel_width - 1, kernel_width - 1])
    else:
        kernel = np.zeros([kernel_width, kernel_width])

    if kernel_type == 'box':
        assert phase == 0.5, 'Box filter is always half-phased'
        kernel[:] = 1.0 / (kernel_width * kernel_width)

    elif kernel_type == 'gauss':
        assert sigma, 'sigma is not specified'
        assert phase != 0.5, 'phase 1/2 for gauss not implemented'

        center = (kernel_width + 1.0) / 2.0
        sigma_sq =  sigma * sigma
        for i in range(1, kernel.shape[0] + 1):
            for j in range(1, kernel.shape[1] + 1):
                di = (i - center) / 2.0
                dj = (j - center) / 2.0
                kernel[i - 1][j - 1] = np.exp(-(di * di + dj * dj) / (2 * sigma_sq))
                kernel[i - 1][j - 1] = kernel[i - 1][j - 1] / (2. * np.pi * sigma_sq)

    elif kernel_type == 'lanczos':
        assert support, 'support is not specified'

        center = (kernel_width + 1) / 2.
        for i in range(1, kernel.shape[0] + 1):
            for j in range(1, kernel.shape[1] + 1):
                if phase == 0.5:
                    di = abs(i + 0.5 - center) / factor
                    dj = abs(j + 0.5 - center) / factor
                else:
                    di = abs(i - center) / factor
                    dj = abs(j - center) / factor

                val = 1
                if di != 0:
                    val = val * support * np.sin(np.pi * di) * np.sin(np.pi * di / support)
                    val = val / (np.pi * np.pi * di * di)
                if dj != 0:
                    val = val * support * np.sin(np.pi * dj) * np.sin(np.pi * dj / support)
                    val = val / (np.pi * np.pi * dj * dj)

                kernel[i - 1][j - 1] = val

    else:
        raise NotImplementedError('kernel type [%s] is not implemented' % kernel_type)

    kernel /= kernel.sum()

    return kernel


class Downsampler(nn.Module):
    """Downsample input tensor with given downsample kernel type.
       Ref: http://www.realitypixels.com/turk/computergraphics/ResamplingFilters.pdf
    """
    def __init__(self, n_planes, factor, kernel_type, phase=0, kernel_width=None, 
                 support=None, sigma=None, preserve_size=False):
        super(Downsampler, self).__init__()

        assert phase in [0, 0.5], 'phase should be 0 or 0.5'

        if kernel_type == 'lanczos2':
            support = 2
            kernel_width = 4 * factor + 1
            kernel_type_ = 'lanczos'

        elif kernel_type == 'lanczos3':
            support = 3
            kernel_width = 6 * factor + 1
            kernel_type_ = 'lanczos'

        elif kernel_type == 'gauss12':
            kernel_width = 7
            sigma = 1/2
            kernel_type_ = 'gauss'

        elif kernel_type == 'gauss1sq2':
            kernel_width = 9
            sigma = 1.0 / np.sqrt(2)
            kernel_type_ = 'gauss'

        elif kernel_type in ['lanczos', 'gauss', 'box']:
            kernel_type_ = kernel_type

        else:
            raise NotImplementedError('kernel type [%s] is not implemented' % kernel_type)

        # note that `kernel width` will be different to actual size for phase = 1/2
        self.kernel = get_kernel(factor, kernel_type_, phase, kernel_width, support=support, sigma=sigma)

        downsampler = nn.Conv2d(n_planes, n_planes, kernel_size=self.kernel.shape, stride=factor, padding=0)
        downsampler.weight.data.requires_grad_(False).fill_(0)
        downsampler.bias.data.requires_grad_(False).fill_(0)
        downsampler.weight.data.add_(torch.from_numpy(self.kernel).float())

        self.downsampler = downsampler

        if preserve_size:
            if  self.kernel.shape[0] % 2 == 1: 
                pad = int((self.kernel.shape[0] - 1) / 2.0)
            else:
                pad = int((self.kernel.shape[0] - factor) / 2.0)

            self.padding = nn.ReplicationPad2d(pad)

        self.preserve_size = preserve_size

    def forward(self, x):
        if self.preserve_size:
            x = self.padding(x)
        return self.downsampler(x)


class ConvBlock(nn.Module):
    def __init__(self, inplanes, planes, 
                 kernel_size, stride=1, bias=True, padding_mode='reflect', downsample_mode='stride'):
        super(ConvBlock, self).__init__()

        padding_size = int((kernel_size - 1) / 2)

        if stride != 1 and downsample_mode != 'stride':
            padding_size += 1

            if downsample_mode == 'avg':
                downsampler = nn.AvgPool2d(stride, stride)
            elif downsample_mode == 'max':
                downsampler = nn.MaxPool2d(stride, stride)
            elif downsample_mode  in ['lanczos2', 'lanczos3']:
                downsampler = Downsampler(n_planes=planes, factor=stride, kernel_type=downsample_mode, phase=0.5, preserve_size=True)
            else:
                raise NotImplementedError('downsample [%s] is not implemented' % downsample_mode)

            stride = 1
        else:
            downsampler = None

        padder = TemporalPad2d(padding_type=padding_mode, padding_size=padding_size)()

        convolver = nn.Conv2d(inplanes, planes, kernel_size, stride, bias=bias)

        layers = filter(lambda x: x is not None, [padder, convolver, downsampler])

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_block(x)


class Concat(nn.Module):
    def __init__(self, dim, *args):
        super(Concat, self).__init__()
        self.dim = dim

        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, input):
        inputs = []
        for module in self._modules.values():
            inputs.append(module(input))

        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]        

        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(np.array(inputs_shapes3) == min(inputs_shapes3)):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)

            inputs_ = []
            for inp in inputs: 
                diff2 = (inp.size(2) - target_shape2) // 2 
                diff3 = (inp.size(3) - target_shape3) // 2 
                inputs_.append(inp[:, :, diff2: diff2 + target_shape2, diff3:diff3 + target_shape3])

        return torch.cat(inputs_, dim=self.dim)

    def __len__(self):
        return len(self._modules)


##############################################################################
# ResNet Network Classes
##############################################################################


# class ResidualBlock(nn.Module):
#     """Residual Basic Block"""

#     def __init__(self, inplanes, planes, stride=1, norm_layer=None, act_func=None, 
#                  downsample=None, residual=True):
#         super(ResidualBlock, self).__init__()
#         norm_layer = Normalize(norm_layer)
#         act_func = NonLinear(act_func)
#         layers = [ resnet.conv3x3(inplanes, planes, stride),
#                    norm_layer(planes),
#                    act_func(),
#                    resnet.conv3x3(planes, planes),
#                    norm_layer(planes) ]
#         self.layers = nn.Sequential(*layers)
#         self.downsample = downsample
#         self.residual = residual

#     def _match_shape(self, x1, size):
#         if (x1.size(2) != size[2]) or (x1.size(3) != size[3]):
#             dh = (x1.size(2) - size[2]) // 2
#             dw = (x1.size(3) - size[3]) // 2
#             x1 = x1[..., dh:dh+size[2], dw:dw+size[3]]
#         return x1

#     def forward(self, x):
#         identity = x

#         x = self.layers(x)
#         if self.downsample:
#             x = self.downsample(x)

#         identity = self._match_shape(identity, x.size())
#         if self.residual:
#             x += identity

#         return x


class ResNetBlock(nn.Module):
    """Residual Basic Block"""

    def __init__(self, inplanes, planes, stride=1, norm_layer=None, act_func=None):
        super(ResNetBlock, self).__init__()
        norm_layer = Normalize(norm_layer)
        act_func = NonLinear(act_func)
        layers = [ resnet.conv3x3(inplanes, planes, stride),
                   norm_layer(planes),
                   act_func(),
                   resnet.conv3x3(planes, planes),
                   norm_layer(planes) ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ResidualConnectionBlock(nn.Module):
    def __init__(self, submodule=None, downsample=None, residual=True):
        super(ResidualConnectionBlock, self).__init__()
        self.submodule = submodule
        self.downsample = downsample
        self.residual = residual

    def _match_shape(self, x1, size):
        if (x1.size(2) != size[2]) or (x1.size(3) != size[3]):
            dh = (x1.size(2) - size[2]) // 2
            dw = (x1.size(3) - size[3]) // 2
            x1 = x1[..., dh:dh+size[2], dw:dw+size[3]]
        return x1

    def forward(self, x):
        identity = x

        x = self.submodule(x)
        if self.downsample:
            x = self.downsample(x)

        identity = self._match_shape(identity, x.size())
        if self.residual:
            x += identity

        return x


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channels, out_channels, channels, 
                 padding_mode='reflect', norm_layer='batch', 
                 act_func='leakyrelu', final_act_func='sigmoid', residual=False):
        super(ResNet, self).__init__()

        self.prenet = nn.Sequential(*[
            ConvBlock(in_channels, channels, 3, stride=1, padding_mode=padding_mode),
            NonLinear(act_func)()])

        blocks = [
            # block(channels, channels, norm_layer=norm_layer, act_func=act_func, residual=residual)
            ResidualConnectionBlock(block(channels, channels, norm_layer=norm_layer, act_func=act_func), residual=residual)
            for _ in range(num_blocks)]
        blocks += [
            resnet.conv3x3(channels, channels, stride=1), Normalize(norm_layer)(channels)]
        self.blocks = nn.Sequential(*blocks)

        self.postnet = nn.Sequential(*[
            ConvBlock(channels, out_channels, 3, stride=1, padding_mode=padding_mode),
            NonLinear(final_act_func)()])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.prenet(x)
        x = self.blocks(x)
        x = self.postnet(x)

        return x


##############################################################################
# Skip Connection Network Classes
##############################################################################


class MultiConvBlock(nn.Module):
    def __init__(self, inplanes, planes, n_layers=2, 
                 norm_layer=None, act_func=None, bias=True, padding_mode='reflect'):
        super(MultiConvBlock, self).__init__()

        layers = [
            ConvBlock(inplanes, planes, 3, bias=bias, padding_mode=padding_mode),
            Normalize(norm_layer)(planes),
            NonLinear(act_func)(),
        ]

        for _ in range(n_layers-1):
            layers += [
                ConvBlock(planes, planes, 3, bias=bias, padding_mode=padding_mode),
                Normalize(norm_layer)(planes),
                NonLinear(act_func)()
            ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# class UNetDown(nn.Module):
#     def __init__(self, inplanes, planes, 
#                  norm_layer=None, act_func=None, bias=True, padding_mode='reflect'):
#         super(UNetDown, self).__init__()

#         self.conv = MultiConvBlock(inplanes, planes, norm_layer=norm_layer, act_func=act_func, 
#                              bias=bias, padding_mode=padding_mode)
#         self.down = nn.MaxPool2d(2, 2)

#     def forward(self, x):
#         return self.down(self.conv(x))


# class UNetUp(nn.Module):
#     def __init__(self, inplanes, planes, upsample_mode='deconv', 
#                  norm_layer=None, act_func=None, bias=True, padding_mode='reflect'):
#         super(UNetUp, self).__init__()

#         if upsample_mode == 'deconv':
#             self.up = nn.Sequential(
#                 nn.ConvTranspose2d(inplanes, planes, 4, stride=2, padding=1))
#         elif upsample_mode=='bilinear' or upsample_mode=='nearest':
#             self.up = nn.Sequential(
#                 nn.Upsample(scale_factor=2, mode=upsample_mode),
#                 ConvBlock(inplanes, planes, 3, bias=bias, padding_mode=padding_mode))
#         self.conv = MultiConvBlock(planes * 2, planes, norm_layer=norm_layer, act_func=act_func, 
#                              bias=bias, padding_mode=padding_mode)

#     def _match_shape(self, x1, size):
#         if (x1.size(2) != size[2]) or (x1.size(3) != size[3]):
#             dh = (x1.size(2) - size[2]) // 2
#             dw = (x1.size(3) - size[3]) // 2
#             x1 = x1[..., dh:dh+size[2], dw:dw+size[3]]
#         return x1

#     def forward(self, x, sup):
#         x = self.up(x)
#         sup = self._match_shape(sup, x.size())
#         x = self.conv(torch.cat([x, sup], 1))
#         return x


# class UNet(nn.Module):
#     def __init__(self, in_channels=3, out_channels=3, feature_scale=4, upsample_mode='deconv', 
#                        padding_mode='zero', norm_layer='instance', act_func='relu', use_sigmoid=True, bias=True):
#         super(UNet, self).__init__()

#         self.feature_scale = feature_scale

#         filters = [64, 128, 256, 512, 1024]
#         filters = [x // self.feature_scale for x in filters]

#         self.prenet = MultiConvBlock(in_channels, filters[0], norm_layer=norm_layer, act_func=act_func, 
#                                bias=bias, padding_mode=padding_mode)
        
#         self.down1 = UNetDown(filters[0], filters[1], norm_layer=norm_layer, act_func=act_func, 
#                               bias=bias, padding_mode=padding_mode)
#         self.down2 = UNetDown(filters[1], filters[2], norm_layer=norm_layer, act_func=act_func, 
#                               bias=bias, padding_mode=padding_mode)
#         self.down3 = UNetDown(filters[2], filters[3], norm_layer=norm_layer, act_func=act_func, 
#                               bias=bias, padding_mode=padding_mode)
#         self.down4 = UNetDown(filters[3], filters[4], norm_layer=norm_layer, act_func=act_func, 
#                               bias=bias, padding_mode=padding_mode)

#         self.up4 = UNetUp(filters[4], filters[3], upsample_mode=upsample_mode, 
#                           bias=bias, padding_mode=padding_mode)
#         self.up3 = UNetUp(filters[3], filters[2], upsample_mode=upsample_mode, 
#                           bias=bias, padding_mode=padding_mode)
#         self.up2 = UNetUp(filters[2], filters[1], upsample_mode=upsample_mode, 
#                           bias=bias, padding_mode=padding_mode)
#         self.up1 = UNetUp(filters[1], filters[0], upsample_mode=upsample_mode, 
#                           bias=bias, padding_mode=padding_mode)

#         self.postnet = nn.Sequential(
#             ConvBlock(filters[0], out_channels, 1, bias=bias, padding_mode=padding_mode),
#             NonLinear('sigmoid' if use_sigmoid else 'none')())

#     def forward(self, x):
#         down0 = self.prenet(x)
#         down1 = self.down1(down0)
#         down2 = self.down2(down1)
#         down3 = self.down3(down2)
#         down4 = self.down4(down3)

#         up4 = self.up4(down4, down3)
#         up3 = self.up3(up4, down2)
#         up2 = self.up2(up3, down1)
#         up1 = self.up1(up2, down0)
#         up0 = self.postnet(up1)

#         return up0


class SkipDown(nn.Module):
    def __init__(self, inplanes, planes, downsample_mode='stride',
                 norm_layer=None, act_func=None, bias=True, padding_mode='reflect'):
        super(SkipDown, self).__init__()

        self.conv = MultiConvBlock(inplanes, planes, norm_layer=norm_layer, act_func=act_func, 
                             bias=bias, padding_mode=padding_mode)
        self.down = nn.MaxPool2d(2, 2)

    def forward(self, x):
        return self.down(self.conv(x))


class SkipUp(nn.Module):
    def __init__(self, inplanes, planes, upsample_mode='deconv', 
                 norm_layer=None, act_func=None, bias=True, padding_mode='reflect'):
        super(SkipUp, self).__init__()

        if upsample_mode == 'deconv':
            self.up = nn.Sequential(
                nn.ConvTranspose2d(inplanes, planes, 4, stride=2, padding=1))
        elif upsample_mode=='bilinear' or upsample_mode=='nearest':
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode=upsample_mode),
                ConvBlock(inplanes, planes, 3, bias=bias, padding_mode=padding_mode))
        self.conv = MultiConvBlock(planes, planes, norm_layer=norm_layer, act_func=act_func, 
                             bias=bias, padding_mode=padding_mode)

    def forward(self, x):
        return self.conv(self.up(x))


class SkipConnectionBlock(nn.Module):

    def __init__(self, in_channels, out_channels, channels, skip_channels, mode='skip',
                 submodule=None, upsample_mode='deconv', downsample_mode='stride', 
                 kernel_size=3, skip_kernel_size=1, norm_layer=None, act_func=None, 
                 bias=True, padding_mode='reflect', use_conv1x1=True):
        super(SkipConnectionBlock, self).__init__()

        down = SkipDown(in_channels, channels, downsample_mode=downsample_mode, norm_layer=norm_layer, act_func=act_func, 
                        bias=bias, padding_mode=padding_mode)
        up = SkipUp(channels, channels, upsample_mode=upsample_mode, norm_layer=norm_layer, act_func=act_func, 
                    bias=bias, padding_mode=padding_mode)

        # deeper = nn.Sequential(*[down, submodule, up])
        deeper = nn.Sequential(*filter(lambda x: x is not None, [down, submodule, up]))

        if mode == 'skip':
            skip = nn.Sequential(
                ConvBlock(in_channels, skip_channels, skip_kernel_size, bias=bias, padding_mode=padding_mode),
                Normalize(norm_layer)(skip_channels),
                NonLinear(act_func)()
            )
        elif mode == 'unet':
            skip_channels = in_channels
            skip = nn.Sequential(nn.Identity())

        blocks = [ Concat(1, skip, deeper) ]

        blocks += [
            ConvBlock(channels+skip_channels, channels, kernel_size, bias=bias, padding_mode=padding_mode),
            Normalize(norm_layer)(channels),
            NonLinear(act_func)() ]
        if use_conv1x1:
            blocks += [
                ConvBlock(channels, out_channels, 1, bias=bias, padding_mode=padding_mode),
                Normalize(norm_layer)(out_channels),
                NonLinear(act_func)() ]

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class SkipNet(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=4, nf=16, mode='skip',
                 norm_layer=None, use_sigmoid=True, upsample_mode='deconv', downsample_mode='stride'):
        super(SkipNet, self).__init__()

        block = SkipConnectionBlock(nf * 8, nf * 8, nf * 8, skip_channels=skip_channels, mode=mode,
                                    submodule=None, upsample_mode=upsample_mode, downsample_mode=downsample_mode, norm_layer=norm_layer)
        block = SkipConnectionBlock(nf * 4, nf * 4, nf * 8, skip_channels=skip_channels, mode=mode,
                                    submodule=block, upsample_mode=upsample_mode, downsample_mode=downsample_mode, norm_layer=norm_layer)
        block = SkipConnectionBlock(nf * 2, nf * 2, nf * 4, skip_channels=skip_channels, mode=mode,
                                    submodule=block, upsample_mode=upsample_mode, downsample_mode=downsample_mode, norm_layer=norm_layer)
        block = SkipConnectionBlock(nf * 1, nf * 1, nf * 2, skip_channels=skip_channels, mode=mode,
                                    submodule=block, upsample_mode=upsample_mode, downsample_mode=downsample_mode, norm_layer=norm_layer)
        block = SkipConnectionBlock(in_channels, nf * 1, nf * 1, skip_channels=skip_channels, mode=mode,
                                    submodule=block, upsample_mode=upsample_mode, downsample_mode=downsample_mode, norm_layer=norm_layer)

        self.block = block

        self.postnet = nn.Sequential(
            ConvBlock(nf * 1, out_channels, 1),
            NonLinear('sigmoid' if use_sigmoid else 'none')())

    def forward(self, x):
        return self.postnet(self.block(x))


##############################################################################
# DCGAN Network Classes
##############################################################################


class DCGAN(nn.Module):
    def __init__(self, latent_dim=128, ngf=32, img_size=64, channels=3, 
                 use_conv_trans=True, upsample_mode='bilinear', final_activation='sigmoid'):
        super(DCGAN, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, ngf * self.init_size ** 2))

        conv_blocks = [ nn.BatchNorm2d(128) ]

        for _ in range(2):
            if use_conv_trans:
                conv_blocks += [ nn.ConvTranspose2d(ngf, ngf, kernel_size=4, stride=2, padding=1, bias=False),
                                 nn.BatchNorm2d(ngf),
                                 nn.LeakyReLU(True) ]
            else:
                conv_blocks += [ nn.Upsample(scale_factor=2, mode=upsample_mode),
                                 nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1),
                                 nn.BatchNorm2d(ngf),
                                 nn.LeakyReLU(True) ]

        conv_blocks += [ nn.Conv2d(ngf, channels, kernel_size=3, stride=1, padding=1) ]
        
        if final_activation == 'sigmoid':
            conv_blocks += [ nn.Sigmoid() ]
        elif final_activation == 'tanh':
            conv_blocks += [ nn.Tanh() ]

        self.conv_blocks = nn.Sequential(*conv_blocks)

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


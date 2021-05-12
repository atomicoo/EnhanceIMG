import torch
import torch.nn as nn


class DCGAN(nn.Module):
    def __init__(self, latent_dim=128, ngf=32, img_size=64, channels=3, 
                 use_conv_trans=True, upsample_mode='bilinear', final_activation='sigmoid'):
        super(DCGAN, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, ngf * self.init_size ** 2))

        conv_blocks = [ nn.BatchNorm2d(128) ]

        for _ in range(2):
            if use_conv_trans:
                conv_blocks += [ nn.ConvTranspose2d(ngf, ngf),
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

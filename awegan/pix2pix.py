"""Ref: https://github.com/phillipi/pix2pix"""

import argparse
import os
import os.path as osp
import time
import random
from glob import glob
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms.transforms import Resize
from torchvision.utils import save_image
from PIL import Image

manualSeed = 2021
manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class Generator(nn.Module):
    def __init__(self, d=64):
        super(Generator, self).__init__()

        def conv_block(in_channels, out_channels, stride=2, normalize=True):
            return nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.Conv2d(in_channels, out_channels, 4, stride, 1),
                nn.BatchNorm2d(out_channels) if normalize else nn.Identity(),
            )

        def deconv_block(in_channels, out_channels, stride=2, normalize=True):
            return nn.Sequential(
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels, out_channels, 4, stride, 1),
                nn.BatchNorm2d(out_channels) if normalize else nn.Identity(),
            )

        # Unet encoder
        self.conv1 = nn.Conv2d(3, d, 4, 2, 1)
        self.conv2 = conv_block(d * 1, d * 2)
        self.conv3 = conv_block(d * 2, d * 4)
        self.conv4 = conv_block(d * 4, d * 8)
        self.conv5 = conv_block(d * 8, d * 8)
        self.conv6 = conv_block(d * 8, d * 8)
        self.conv7 = conv_block(d * 8, d * 8)
        self.conv8 = conv_block(d * 8, d * 8, normalize=False)

        # Unet decoder
        self.deconv1 = deconv_block(d * 8 , d * 8)
        self.deconv2 = deconv_block(d * 16, d * 8)
        self.deconv3 = deconv_block(d * 16, d * 8)
        self.deconv4 = deconv_block(d * 16, d * 8)
        self.deconv5 = deconv_block(d * 16, d * 4)
        self.deconv6 = deconv_block(d * 8 , d * 2)
        self.deconv7 = deconv_block(d * 4 , d * 1)
        self.deconv8 = deconv_block(d * 2 , 3, normalize=False)

    # weight_init
    def weight_init(self, mean=0.0, std=0.02):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        e1 = self.conv1(input)
        e2 = self.conv2(e1)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        e5 = self.conv5(e4)
        e6 = self.conv6(e5)
        e7 = self.conv7(e6)
        e8 = self.conv8(e7)

        d1 = F.dropout(self.deconv1(e8), 0.5, training=True)
        d1 = torch.cat([d1, e7], 1)
        d2 = F.dropout(self.deconv2(d1), 0.5, training=True)
        d2 = torch.cat([d2, e6], 1)
        d3 = F.dropout(self.deconv3(d2), 0.5, training=True)
        d3 = torch.cat([d3, e5], 1)
        d4 = self.deconv4(d3)
        d4 = torch.cat([d4, e4], 1)
        d5 = self.deconv5(d4)
        d5 = torch.cat([d5, e3], 1)
        d6 = self.deconv6(d5)
        d6 = torch.cat([d6, e2], 1)
        d7 = self.deconv7(d6)
        d7 = torch.cat([d7, e1], 1)
        d8 = self.deconv8(d7)
        output = torch.tanh(d8)

        return output

class Discriminator(nn.Module):
    # initializers
    def __init__(self, d=64):
        super(Discriminator, self).__init__()

        def conv_block(in_channels, out_channels, stride=2, normalize=True):
            return nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.Conv2d(in_channels, out_channels, 4, stride, 1),
                nn.BatchNorm2d(out_channels) if normalize else nn.Identity(),
            )

        self.conv1 = nn.Conv2d(6, d, 4, 2, 1)
        self.conv2 = conv_block(d * 1, d * 2)
        self.conv3 = conv_block(d * 2, d * 4)
        self.conv4 = conv_block(d * 4, d * 8, stride=1)
        self.conv5 = conv_block(d * 8, 1, stride=1, normalize=False)

    # weight_init
    def weight_init(self, mean=0.0, std=0.02):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        output = torch.sigmoid(x)

        return output


def create_data_loader(path, subfolder, transform, batch_size, shuffle=True, num_workers=8):
    """Create train/test data_loader
    Dataset: https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/
    """
    dset = datasets.ImageFolder(path, transform)
    idx = dset.class_to_idx[subfolder]

    cnt = 0
    for _ in range(len(dset)):
        if dset.imgs[cnt][1] != idx:
            del dset.imgs[cnt]
            cnt -= 1
        cnt += 1

    return DataLoader(dset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def get_last_chkpt_path(logdir):
    """Returns the last checkpoint file name in the given log dir path."""
    checkpoints = glob(osp.join(logdir, '*.pth'))
    checkpoints.sort()
    if len(checkpoints) == 0:
        return None
    return checkpoints[-1]


def imresize(img, size):
    img = ((img+1.0)/2.0*255.0).clip(0, 255).transpose(1, 2, 0).astype(np.uint8)
    img = np.array(Image.fromarray(img).resize(size))
    img = ((img.astype(np.float32)/255.0)*2.0-1.0).clip(-1.0, 1.0).transpose(2, 0, 1)
    return img

def imgs_resize(imgs, resize_scale=286):
    outputs = torch.FloatTensor(imgs.size()[0], imgs.size()[1], resize_scale, resize_scale)
    for i in range(imgs.size(0)):
        img = imresize(imgs[i].numpy(), [resize_scale, resize_scale])
        outputs[i] = torch.FloatTensor((img.transpose(2, 0, 1).astype(np.float32).reshape(-1, imgs.size()[1], resize_scale, resize_scale) - 127.5) / 127.5)

    return outputs

def random_crop(imgs1, imgs2, crop_size=256):
    outputs1 = torch.FloatTensor(imgs1.size(0), imgs1.size(1), crop_size, crop_size)
    outputs2 = torch.FloatTensor(imgs2.size(0), imgs2.size(1), crop_size, crop_size)
    for i in range(imgs1.size(0)):
        rand1 = random.randint(0, imgs1.size(2) - crop_size) # np.random.randint(0, imgs1.size(2) - crop_size)
        rand2 = random.randint(0, imgs1.size(3) - crop_size) # np.random.randint(0, imgs2.size(3) - crop_size)
        outputs1[i] = imgs1[i, :, rand1: crop_size + rand1, rand2: crop_size + rand2]
        outputs2[i] = imgs2[i, :, rand1: crop_size + rand1, rand2: crop_size + rand2]

    return outputs1, outputs2

def random_fliplr(imgs1, imgs2):
    outputs1 = torch.FloatTensor(imgs1.size())
    outputs2 = torch.FloatTensor(imgs2.size())
    for i in range(imgs1.size(0)):
        if torch.rand(1)[0] < 0.5:
            img1 = torch.FloatTensor(
                (np.fliplr(imgs1[i].numpy().transpose(1, 2, 0)).transpose(2, 0, 1).reshape(-1, imgs1.size(1), imgs1.size(2), imgs1.size(3)) + 1) / 2)
            outputs1[i] = (img1 - 0.5) / 0.5
            img2 = torch.FloatTensor(
                (np.fliplr(imgs2[i].numpy().transpose(1, 2, 0)).transpose(2, 0, 1).reshape(-1, imgs2.size(1), imgs2.size(2), imgs2.size(3)) + 1) / 2)
            outputs2[i] = (img2 - 0.5) / 0.5
        else:
            outputs1[i] = imgs1[i]
            outputs2[i] = imgs2[i]

    return outputs1, outputs2


def train(opt):

    os.makedirs(opt.outputs_dir, exist_ok=True)
    os.makedirs(opt.ckpts_dir, exist_ok=True)

    device = torch.device("cuda:0" if (torch.cuda.is_available() and opt.n_gpu > 0) else "cpu")

    # Model definition
    generator = Generator(opt.hidden_dim).to(device)
    discriminator = Discriminator(opt.hidden_dim).to(device)

    if (device.type == 'cuda') and (opt.n_gpu > 1):
        generator = nn.DataParallel(generator, list(range(opt.n_gpu)))
        discriminator = nn.DataParallel(discriminator, list(range(opt.n_gpu)))

    generator.weight_init()
    discriminator.weight_init()

    # Loss function
    criterion_gan = nn.BCELoss().to(device)
    criterion_cycle = nn.L1Loss().to(device)

    # Adam optimizer
    g_optimizer = optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

    checkpoint = get_last_chkpt_path(opt.ckpts_dir)
    if opt.warmup and checkpoint:
        checkpoint = torch.load(checkpoint)
        generator.load_state_dict(checkpoint['model_g'])
        discriminator.load_state_dict(checkpoint['model_d'])
        g_optimizer.load_state_dict(checkpoint['optimizer_g'])
        d_optimizer.load_state_dict(checkpoint['optimizer_d'])

    # Data loader
    transform = transforms.Compose([
        transforms.Resize(size=opt.resize_scale),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    data_root = osp.join(opt.data_dir, opt.dataset)
    train_loader = create_data_loader(data_root, "train", transform, 
                                      batch_size=opt.batch_size, num_workers=opt.n_cpu)
    test_loader = create_data_loader(data_root, "test", transform, 
                                     batch_size=opt.batch_size, num_workers=opt.n_cpu)

    test, _ = iter(test_loader).next()
    x_, y_ = test.chunk(2, axis=-1)
    x_, y_ = random_crop(x_, y_, opt.img_size)
    x_, y_ = x_.to(device), y_.to(device)
    gr_sz = generator(x_).size()
    dr_sz = discriminator(x_, y_).size()

    # Training history
    train_hist = {}
    train_hist['d_losses'] = []
    train_hist['g_losses'] = []
    train_hist['epoch_time'] = []
    train_hist['total_time'] = []

    Tensor = torch.cuda.FloatTensor if (device.type == 'cuda') else torch.FloatTensor

    # Start training
    since = time.time()
    for epoch in range(1, opt.n_epochs+1):
        d_losses, g_losses = [], []

        epoch_since = time.time()
        for idx, (batch, _) in enumerate(train_loader):
            if opt.inverse_order:
                y_, x_ = batch.chunk(2, axis=-1)
            else:
                x_, y_ = batch.chunk(2, axis=-1)

            # Data augmentation
            # if opt.augment:
            # x_ = imgs_resize(x_, opt.resize_scale)
            # y_ = imgs_resize(y_, opt.resize_scale)
            x_, y_ = random_crop(x_, y_, opt.img_size)
            x_, y_ = random_fliplr(x_, y_)

            x_, y_ = Variable(x_.to(device)), Variable(y_.to(device))

            # Train discriminator
            discriminator.zero_grad()

            size = [batch.size(0), dr_sz[1], dr_sz[2], dr_sz[3]]
            real = Variable(Tensor(size=size).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(size=size).fill_(0.0), requires_grad=False)

            d_result = discriminator(x_, y_)
            real_loss = criterion_gan(d_result, real)
            g_result = generator(x_)
            d_result = discriminator(x_, g_result)
            fake_loss = criterion_gan(d_result, fake)

            d_loss = (real_loss + fake_loss) * 0.5
            d_loss.backward()
            d_optimizer.step()

            train_hist['d_losses'].append(d_loss.item())
            d_losses.append(d_loss.item())

            # Train generator
            generator.zero_grad()

            g_result = generator(x_)
            d_result = discriminator(x_, g_result)
            g_loss = criterion_gan(d_result, real) + opt.l1_lambda * criterion_cycle(g_result, y_)
            g_loss.backward()
            g_optimizer.step()

            train_hist['g_losses'].append(g_loss.item())
            g_losses.append(g_loss.item())

            print('[Epoch %d/%d] [Batch %d/%d] - Elapsed: %.2fs, D loss: %.3f, G loss: %.3f' % 
                    (epoch, opt.n_epochs, idx, len(train_loader), time.time() - epoch_since, 
                    torch.FloatTensor(d_losses).mean(), torch.FloatTensor(g_losses).mean()))

        train_hist['epoch_time'].append(epoch_since)

        if epoch % opt.sample_interval == 0:
            ipath = osp.join(opt.outputs_dir, f'{time.strftime("%Y-%m-%d")}_{epoch:03d}.png')
            save_image(g_result.data[:1], ipath, nrow=1, normalize=True)
        
        if epoch % opt.save_interval == 0:
            torch.save({
                "model_g": generator.state_dict(),
                "model_d": discriminator.state_dict(),
                "optimizer_g": g_optimizer.state_dict(),
                "optimizer_d": d_optimizer.state_dict()
            }, osp.join(opt.ckpts_dir, f'{time.strftime("%Y-%m-%d")}_chkpt_epoch{epoch:03d}.pth'))

    total_time = time.time() - since
    train_hist['total_time'].append(total_time)

    print("Avg elapsed time per epoch: %.2fs (total %d epochs elapsed %.2fs)" % 
            (torch.FloatTensor(train_hist['epoch_time']).mean(), opt.n_epochs, total_time))
    print("Training finished!\n")


def test(opt):
    os.makedirs(opt.ckpts_dir, exist_ok=True)
    os.makedirs(opt.outputs_dir, exist_ok=True)

    device = torch.device("cuda:0" if (torch.cuda.is_available() and opt.n_gpu > 0) else "cpu")

    checkpoint = get_last_chkpt_path(opt.ckpts_dir)
    checkpoint = torch.load(checkpoint, map_location=device)

    # Model definition
    generator = Generator(opt.hidden_dim).to(device)
    generator.load_state_dict(checkpoint['model_g'])

    # Data loader
    transform = transforms.Compose([
        # transforms.Resize(size=opt.resize_scale),
        # transforms.Resize(size=(2*384, 2*512*2)),
        # transforms.CenterCrop(opt.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    data_root = osp.join(opt.data_dir, opt.dataset)
    dataloader = create_data_loader(data_root, "test", transform, 
                                    batch_size=opt.batch_size, num_workers=opt.n_cpu)

    for idx, (batch, _) in enumerate(dataloader):
        if opt.inverse_order:
            y_, x_ = batch.chunk(2, axis=-1)
        else:
            x_, y_ = batch.chunk(2, axis=-1)
        # x_, y_ = random_crop(x_, y_, opt.img_size)

        g_result = generator(x_)

        ipath = osp.join(opt.outputs_dir, f'{time.strftime("%Y-%m-%d")}_batch{idx:03d}.png')
        g_result = torch.cat((x_.data[:4], g_result.data[:4]), 0)
        save_image(g_result, ipath, nrow=4, padding=0, normalize=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--warmup", type=bool, default=True, help='warm start from a pretrained model')
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--beta1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--beta2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
    parser.add_argument('--l1_lambda', type=float, default=100, help='lambda for L1 loss')
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--n_gpu", type=int, default=1, help="number of gpu to use during training model")
    parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
    parser.add_argument("--hidden_dim", type=int, default=64, help="dimensionality of the feature space")
    parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
    # parser.add_argument('--augment', type=bool, default=True, help='data augment True or False')
    # parser.add_argument('--crop_size', type=int, default=256, help='crop size (0 is false)')
    parser.add_argument('--resize_scale', type=int, default=286, help='resize scale (0 is false)')
    # parser.add_argument('--fliplr', type=bool, default=True, help='random fliplr True or False')
    parser.add_argument("--data_dir", type=str, default="data/pix2pix", help="data root directory")
    parser.add_argument("--dataset", type=str, default="facades", help="dataset's name")
    parser.add_argument("--outputs_dir", type=str, default="results", help="output wave file directory")
    parser.add_argument("--ckpts_dir", type=str, default="ckpts", help="saved model file directory")
    parser.add_argument('--inverse_order', type=bool, default=True, help='0: [input, target], 1 - [target, input]')
    parser.add_argument("--sample_interval", type=int, default=25, help="interval between image sampling")
    parser.add_argument("--save_interval", type=int, default=50, help="interval between model saving")
    opt = parser.parse_args()
    print(opt)

    train(opt)
    # test(opt)

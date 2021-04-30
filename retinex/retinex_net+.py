"""Ref: https://github.com/aasharma90/RetinexNet_PyTorch"""

import argparse
import os
import os.path as osp
import time
import random
from glob import glob
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image


def get_last_chkpt_path(logdir):
    """Returns the last checkpoint file name in the given log dir path."""
    checkpoints = glob(osp.join(logdir, '*.pth'))
    checkpoints.sort()
    if len(checkpoints) == 0:
        return None
    return checkpoints[-1]


class Processor(nn.Module):
    def __init__(self):
        super(Processor, self).__init__()

    def forward(img):
        img = np.float32(img) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        img = np.ascontiguousarray(img)
        return img

    def inverse(img):
        img = np.squeeze(np.float32(img), axis=0)
        img = np.transpose(img, (1, 2, 0))
        img = np.clip(img * 255.0, 0, 255.0).astype('uint8')
        img = np.ascontiguousarray(img)
        return img


class PairDataset(Dataset):
    def __init__(self, root, transform=None, augmentation=True, mode='train'):
        super(PairDataset, self).__init__()

        self.name = 'PairDataset'
        self.mode = mode
        self.patch_size = None
        self.augmentation = augmentation

        _transforms = []
        for trans in transform:
            if isinstance(trans, transforms.RandomCrop):
                self.patch_size = trans.size
            else:
                _transforms.append(trans)
        self.transform = transforms.ToTensor() if transform is None else transforms.Compose(_transforms)

        # self.files_A = sorted(glob(os.path.join(root, '%sA' % mode, '*.*')))
        # self.files_B = sorted(glob(os.path.join(root, '%sB' % mode, '*.*')))
        self.files_A = sorted(glob(os.path.join(root, mode, 'low', '*.*')))
        self.files_B = sorted(glob(os.path.join(root, mode, 'high', '*.*')))

        assert len(self.files_A) == len(self.files_B), "Images of A and B must be matched"

    def __getitem__(self, index):
        A_path = self.files_A[index % len(self.files_A)]
        B_path = self.files_B[index % len(self.files_B)]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(A_path.replace("low", "high")).convert('RGB')

        item_A = self.transform(A_img).permute(1, 2, 0).contiguous()
        item_B = self.transform(B_img).permute(1, 2, 0).contiguous()

        assert item_A.size() == item_B.size(), "Size of A and B doesn't matched"
        h, w, c = item_A.size()

        if self.patch_size is not None:
            x = random.randint(0, h - self.patch_size[0])
            y = random.randint(0, w - self.patch_size[1])
            item_A = item_A[x: x + self.patch_size[0], y: y + self.patch_size[1], :]
            item_B = item_B[x: x + self.patch_size[0], y: y + self.patch_size[1], :]

        if self.augmentation:
            item_A = item_A.cpu().numpy()
            item_B = item_B.cpu().numpy()

            # Data augmentation
            # TODO: Realize this with pytorch
            if random.random() < 0.5:
                item_A = np.flipud(item_A)
                item_B= np.flipud(item_B)
            if random.random() < 0.5:
                item_A = np.fliplr(item_A)
                item_B= np.fliplr(item_B)
            rot_type = random.randint(1, 4)
            if random.random() < 0.5:
                item_A = np.rot90(item_A, rot_type)
                item_B= np.rot90(item_B, rot_type)

            item_A = torch.from_numpy(np.ascontiguousarray(item_A))
            item_B = torch.from_numpy(np.ascontiguousarray(item_B))

        item_A = item_A.permute(2, 0, 1).contiguous()
        item_B = item_B.permute(2, 0, 1).contiguous()

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


class DecomNet(nn.Module):
    def __init__(self, channels=64, kernel_size=3):
        super(DecomNet, self).__init__()
        # Shallow feature extraction
        self.netD_conv0 = nn.Conv2d(4, channels, kernel_size * 3,
                                    padding=4, padding_mode='replicate')
        # Activated layers!
        self.netD_convs = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size,
                      padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size,
                      padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size,
                      padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size,
                      padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size,
                      padding=1, padding_mode='replicate'),
            nn.ReLU())
        # Final recon layer
        self.netD_recon = nn.Conv2d(channels, 4, kernel_size,
                                    padding=1, padding_mode='replicate')

    def forward(self, img):
        cmax = torch.max(img, dim=1, keepdim=True)[0]
        img4 = torch.cat((cmax, img), dim=1)
        fs0  = self.netD_conv0(img4)
        fss  = self.netD_convs(fs0)
        out  = self.netD_recon(fss)
        R    = torch.sigmoid(out[:, 0:3, :, :])
        L    = torch.sigmoid(out[:, 3:4, :, :])
        return R, L


class RelightNet(nn.Module):
    def __init__(self, channels=64, kernel_size=3):
        super(RelightNet, self).__init__()

        def conv_relu(channels, kernel_size):
            return nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size, stride=2,
                          padding=1, padding_mode='replicate'),
                nn.ReLU()
            )
        
        def deconv_relu(channels, kernel_size):
            return nn.Sequential(
                nn.Conv2d(channels*2, channels, kernel_size,
                          padding=1, padding_mode='replicate'),
                nn.ReLU()
            )

        self.netR_conv0 = nn.Conv2d(4, channels, kernel_size,
                                    padding=1, padding_mode='replicate')

        self.netR_convs1 = conv_relu(channels, kernel_size)
        self.netR_convs2 = conv_relu(channels, kernel_size)
        self.netR_convs3 = conv_relu(channels, kernel_size)

        self.netR_deconvs1 = deconv_relu(channels, kernel_size)
        self.netR_deconvs2 = deconv_relu(channels, kernel_size)
        self.netR_deconvs3 = deconv_relu(channels, kernel_size)

        self.netR_fusion = nn.Conv2d(channels*3, channels, kernel_size=1,
                                     padding=1, padding_mode='replicate')
        self.netR_output = nn.Conv2d(channels, 1, kernel_size=3, padding=0)

    def forward(self, L, R):
        img = torch.cat((R, L), dim=1)
        fs0 = self.netR_conv0(img)
        fs1 = self.netR_convs1(fs0)
        fs2 = self.netR_convs2(fs1)
        fs3 = self.netR_convs3(fs2)

        fs3_up  = F.interpolate(fs3, size=(fs2.size()[2], fs2.size()[3]))
        fs2p    = self.netR_deconvs1(torch.cat((fs3_up, fs2), dim=1))
        fs2p_up = F.interpolate(fs2p, size=(fs1.size()[2], fs1.size()[3]))
        fs1p    = self.netR_deconvs2(torch.cat((fs2p_up, fs1), dim=1))
        fs1p_up = F.interpolate(fs1p, size=(fs0.size()[2], fs0.size()[3]))
        fs0p    = self.netR_deconvs3(torch.cat((fs1p_up, fs0), dim=1))

        fs2p_rs = F.interpolate(fs2p, size=(R.size()[2], R.size()[3]))
        fs1p_rs = F.interpolate(fs1p, size=(R.size()[2], R.size()[3]))
        fs_all  = torch.cat((fs2p_rs, fs1p_rs, fs0p), dim=1)
        fs_fus  = self.netR_fusion(fs_all)
        output  = self.netR_output(fs_fus)
        return output


class RetinexNet(nn.Module):
    def __init__(self, device, ckpt_dir=None):
        super(RetinexNet, self).__init__()

        self.device = device

        self.DecomNet = DecomNet().to(self.device)
        self.RelightNet = RelightNet().to(self.device)

        if ckpt_dir is not None:
            load_model_status, global_step = self.load(ckpt_dir)
            if load_model_status:
                self.global_step = global_step
                print(f"Model restore (step {global_step}) success!")
            else:
                self.global_step = 0
                print("No pretrained model to restore!")

    def forward(self, input_low, input_high=None):
        input_low = torch.from_numpy(input_low) \
                if isinstance(input_low, np.ndarray) else input_low
        input_high = torch.from_numpy(input_high) \
            if isinstance(input_high, np.ndarray) else input_high

        if input_high is None:
            # Forward DecompNet
            input_low  = torch.FloatTensor(input_low).to(self.device)
            R_low , I_low  = self.DecomNet(input_low)

            # Forward RelightNet
            I_delta = self.RelightNet(I_low, R_low)

            # Other variables
            I_low_3   = torch.cat((I_low, I_low, I_low), dim=1)
            I_delta_3 = torch.cat((I_delta, I_delta, I_delta), dim=1)
        else:
            # Forward DecompNet
            input_low  = Variable(torch.FloatTensor(input_low)).to(self.device)
            input_high = Variable(torch.FloatTensor(input_high)).to(self.device)
            R_low , I_low  = self.DecomNet(input_low)
            R_high, I_high = self.DecomNet(input_high)

            # Forward RelightNet
            I_delta = self.RelightNet(I_low, R_low)

            # Other variables
            I_low_3   = torch.cat((I_low, I_low, I_low), dim=1)
            I_high_3  = torch.cat((I_high, I_high, I_high), dim=1)
            I_delta_3 = torch.cat((I_delta, I_delta, I_delta), dim=1)

            # Compute losses
            self.recon_loss_low  = F.l1_loss(R_low * I_low_3, input_low)
            self.recon_loss_high = F.l1_loss(R_high * I_high_3, input_high)
            self.recon_loss_mutal_low  = F.l1_loss(R_high * I_low_3, input_low)
            self.recon_loss_mutal_high = F.l1_loss(R_low * I_high_3, input_high)
            self.equal_R_loss = F.l1_loss(R_low,  R_high.detach())
            self.relight_loss = F.l1_loss(R_low * I_delta_3, input_high)

            self.Ismooth_loss_low   = self.smooth(I_low, R_low)
            self.Ismooth_loss_high  = self.smooth(I_high, R_high)
            self.Ismooth_loss_delta = self.smooth(I_delta, R_low)

            self.loss_Decom = self.recon_loss_low + \
                            self.recon_loss_high + \
                            0.001 * self.recon_loss_mutal_low + \
                            0.001 * self.recon_loss_mutal_high + \
                            0.1 * self.Ismooth_loss_low + \
                            0.1 * self.Ismooth_loss_high + \
                            0.01 * self.equal_R_loss
            self.loss_Relight = self.relight_loss + \
                                3 * self.Ismooth_loss_delta

        self.output_R_low   = R_low.detach().cpu()
        self.output_I_low   = I_low_3.detach().cpu()
        self.output_I_delta = I_delta_3.detach().cpu()
        self.output_S       = R_low.detach().cpu() * I_delta_3.detach().cpu()

    def gradient(self, input_tensor, direction):
        self.smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2)).to(self.device)
        self.smooth_kernel_y = torch.transpose(self.smooth_kernel_x, 2, 3)

        if direction == "x":
            kernel = self.smooth_kernel_x
        elif direction == "y":
            kernel = self.smooth_kernel_y
        grad_out = torch.abs(F.conv2d(input_tensor, kernel,
                                      stride=1, padding=1))
        return grad_out

    def ave_gradient(self, input_tensor, direction):
        return F.avg_pool2d(self.gradient(input_tensor, direction),
                            kernel_size=3, stride=1, padding=1)

    def smooth(self, input_I, input_R):
        input_R = 0.299*input_R[:, 0, :, :] + 0.587*input_R[:, 1, :, :] + 0.114*input_R[:, 2, :, :]
        input_R = torch.unsqueeze(input_R, dim=1)
        return torch.mean(self.gradient(input_I, "x") * torch.exp(-10 * self.ave_gradient(input_R, "x")) +
                          self.gradient(input_I, "y") * torch.exp(-10 * self.ave_gradient(input_R, "y")))

    def train(self,
              data_root,
              batch_size, patch_size,
              n_epochs,
              lr=0.001, beta1=0.9, beta2=0.999,
              vis_dir="results", ckpt_dir="ckpts",
              eval_interval=10, save_interval=10,
              train_phase=None):

        dataset = PairDataset(
            root=data_root,
            transform=[ transforms.RandomCrop(patch_size),
                        transforms.ToTensor(),
                        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                      ])
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        eval_low_data_fnames = glob(osp.join(data_root, 'eval', 'low', '*.*'))
        eval_low_data_fnames.sort()

        num_batches = len(dataset) // int(batch_size)

        # Create the optimizers
        self.train_op_Decom   = optim.Adam(self.DecomNet.parameters(),
                                           lr=lr[0], betas=(beta1, beta2))
        self.train_op_Relight = optim.Adam(self.RelightNet.parameters(),
                                           lr=lr[0], betas=(beta1, beta2))

        # Initialize a network if its checkpoint is available
        self.train_phase= train_phase
        load_model_status, global_step = self.load(ckpt_dir)
        if load_model_status:
            iter_num    = global_step
            start_epoch = global_step // num_batches
            start_step  = global_step % num_batches
            print("Model restore success!")
        else:
            iter_num    = 0
            start_epoch = 0
            start_step  = 0
            print("No pretrained model to restore!")

        print("Start training for phase %s, with start epoch %d start iter %d : " %
             (self.train_phase, start_epoch, iter_num))

        start_time = time.time()
        image_id   = 0
        for epoch in range(start_epoch, n_epochs):
            self.lr = lr[epoch]
            # Adjust learning rate
            for param_group in self.train_op_Decom.param_groups:
                param_group['lr'] = self.lr
            for param_group in self.train_op_Relight.param_groups:
                param_group['lr'] = self.lr
            
            # Generate training data for a batch
            for batch_id, batch in enumerate(data_loader):
                self.input_low, self.input_high = batch['A'], batch['B']
                
                # Feed-Forward to the network
                self.forward(self.input_low,  self.input_high)

                # Backward to obtain losses
                if self.train_phase == "Decom":
                    self.train_op_Decom.zero_grad()
                    self.loss_Decom.backward()
                    self.train_op_Decom.step()
                    loss = self.loss_Decom.item()
                elif self.train_phase == "Relight":
                    self.train_op_Relight.zero_grad()
                    self.loss_Relight.backward()
                    self.train_op_Relight.step()
                    loss = self.loss_Relight.item()

                print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" \
                      % (train_phase, epoch + 1, batch_id + 1, num_batches, time.time() - start_time, loss))
                iter_num += 1

            # Evaluate the model and save a checkpoint file for it
            if (epoch + 1) % save_interval == 0:
                self.evaluate(epoch + 1, eval_low_data_fnames,
                              vis_dir=vis_dir, train_phase=train_phase)
                self.save(iter_num, ckpt_dir)

        print("Finished training for phase %s." % train_phase)

    def evaluate(self, epoch, eval_low_data_fnames, vis_dir, train_phase):
        print("Evaluating for phase %s / epoch %d..." % (train_phase, epoch))

        for idx in range(len(eval_low_data_fnames)):
            eval_low_img = Image.open(eval_low_data_fnames[idx])
            input_low_eval = Processor.forward(eval_low_img)

            if train_phase == "Decom":
                self.forward(input_low_eval)
                input_S   = input_low_eval
                R_low     = self.output_R_low
                I_low     = self.output_I_low
                cat_image = np.concatenate([input_S, R_low, I_low], axis=3)
            if train_phase == "Relight":
                self.forward(input_low_eval)
                input_S   = input_low_eval
                R_low     = self.output_R_low
                I_low     = self.output_I_low
                I_delta   = self.output_I_delta
                output_S  = self.output_S
                cat_image = np.concatenate([input_S, R_low, I_low, I_delta, output_S], axis=3)

            cat_image = Processor.inverse(cat_image)

            img = Image.fromarray(cat_image)
            filepath = os.path.join(vis_dir, 'eval_%s_%d_%d.png' % (train_phase, epoch, idx + 1))
            img.save(filepath[:-4] + '.jpg')

    def test(self,
             test_low_data_fnames,
             outputs_dir='results',
             ckpts_dir='ckpts',
             save_R_and_L=False):

        # Load the network with a pre-trained checkpoint
        load_model_status, global_step = self.load(ckpts_dir)
        if load_model_status:
            print(f"Model restore (step {global_step}) success!")
        else:
            print("No pretrained model to restore!")
            raise Exception

        # Predict for the test images
        for idx in range(len(test_low_data_fnames)):
            test_img_path = test_low_data_fnames[idx]
            test_img_name = test_img_path.split(os.sep)[-1]

            print('Processing ', test_img_name)
            test_low_img   = Image.open(test_img_path)
            input_low_test = Processor.forward(test_low_img)

            self.forward(input_low_test)

            input_S  = input_low_test
            R_low    = self.output_R_low
            I_low    = self.output_I_low
            I_delta  = self.output_I_delta
            output_S = self.output_S

            # Set `save_R_and_L` switch to True to 
            # also save the reflectance and shading maps
            if save_R_and_L:
                cat_image= np.concatenate([input_S, R_low, I_low, I_delta, output_S], axis=3)
            else:
                cat_image= np.concatenate([input_S, output_S], axis=3)

            cat_image = Processor.inverse(cat_image)

            img = Image.fromarray(cat_image)
            filepath = osp.join(outputs_dir, f'test_{test_img_name}')
            img.save(filepath[:-4] + '.jpg')

    def predict(self,
                test_low_img,
                ckpt_dir='ckpts',
                return_R_and_L=False):

        # Load the network with a pre-trained checkpoint
        load_model_status, global_step = self.load(ckpt_dir)
        if load_model_status:
            print(f"Model restore (step {global_step}) success!")
        else:
            raise FileNotFoundError("No pretrained model to restore!")

        input_low_test = Processor.forward(test_low_img)

        self.forward(input_low_test)

        input_S  = input_low_test
        R_low    = self.output_R_low
        I_low    = self.output_I_low
        I_delta  = self.output_I_delta
        output_S = self.output_S

        cat_image= np.concatenate([input_S, R_low, I_low, I_delta, output_S], axis=3)
        cat_image = Processor.inverse(cat_image)
        input_S, R_low, I_low, I_delta, output_S = np.split(cat_image, 5, axis=1)

        if return_R_and_L:
            return input_S, output_S, (R_low, I_low, I_delta)

        return input_S, output_S, None

    def save(self, global_step, ckpt_dir):
        save_dir = osp.join(ckpt_dir, "RetinexNet")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_fname = f'{time.strftime("%Y-%m-%d")}_chkpt_step{global_step:04d}.pth'
        torch.save({
            "decom": self.DecomNet.state_dict(),
            "relight": self.RelightNet.state_dict(),
            "global_step": global_step
        }, osp.join(save_dir, save_fname))

    def load(self, ckpt_dir):
        load_dir = osp.join(ckpt_dir, "RetinexNet")
        if osp.exists(load_dir):
            print(get_last_chkpt_path(load_dir))
            last_ckpt = get_last_chkpt_path(load_dir)
            if last_ckpt is not None:
                ckpt_dict = torch.load(last_ckpt)
                global_step = ckpt_dict["global_step"]
                self.DecomNet.load_state_dict(ckpt_dict["decom"])
                self.RelightNet.load_state_dict(ckpt_dict["relight"])
                return True, global_step
            else:
                return False, 0
        else:
            return False, 0


def train(args, model):
    lr = args.lr * np.ones([args.n_epochs])
    lr[20:] = lr[0] / 10.0
    lr = np.r_[lr, lr]

    train_low_data_fnames = glob(osp.join(args.data_dir, 'train', 'low', '*.*'))
    train_low_data_fnames.sort()
    train_high_data_fnames = glob(osp.join(args.data_dir, 'train', 'high', '*.*'))
    train_high_data_fnames.sort()
    eval_low_data_fnames = glob(osp.join(args.data_dir, 'eval', 'low', '*.*'))
    eval_low_data_fnames.sort()
    assert len(train_low_data_fnames) == len(train_high_data_fnames)
    print('Number of training data: %d' % len(train_low_data_fnames))

    # Training DecomNet model!
    model.train(args.data_dir,
                batch_size=args.batch_size,
                patch_size=args.patch_size,
                n_epochs=args.n_epochs,
                lr=lr,
                vis_dir=args.outputs_dir,
                ckpt_dir=args.ckpts_dir,
                eval_interval=args.eval_interval,
                save_interval=args.save_interval,
                train_phase="Decom")
    # Training RelightNet model!
    model.train(args.data_dir,
                batch_size=args.batch_size,
                patch_size=args.patch_size,
                n_epochs=args.n_epochs*2,
                lr=lr,
                vis_dir=args.outputs_dir,
                ckpt_dir=args.ckpts_dir,
                eval_interval=args.eval_interval,
                save_interval=args.save_interval,
                train_phase="Relight")


def evaluate(args, model):
    eval_low_data_fnames = glob(osp.join(args.data_dir, 'eval', 'low', '*.*'))
    eval_low_data_fnames.sort()
    print('Number of evaluation images: %d' % len(eval_low_data_fnames))

    model.load(ckpt_dir="ckpts")
    model.evaluate(0, eval_low_data_fnames, "results", "Decom")
    model.evaluate(0, eval_low_data_fnames, "results", "Relight")


def predict(args, model):
    test_low_data_fnames = glob(osp.join(args.data_dir, 'test', 'low', '*.*'))
    test_low_data_fnames.sort()
    print('Number of evaluation images: %d' % len(test_low_data_fnames))

    model.test(test_low_data_fnames,
               outputs_dir=args.outputs_dir,
               ckpts_dir=args.ckpts_dir)

    # test_low_img = Image.open("../Madison.png")
    # input_S, output_S, _ = model.predict(test_low_img)
    # test_high_img = Image.fromarray(np.concatenate([input_S, output_S], axis=1))
    # test_high_img.save("retinexnet.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/LOLdset/', help='root directory of the dataset')
    parser.add_argument("--outputs_dir", type=str, default="results/", help="output wave file directory")
    parser.add_argument("--ckpts_dir", type=str, default="ckpts", help="saved model file directory")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
    parser.add_argument("--patch_size", type=int, default=96, help="size of the patches")
    parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
    parser.add_argument("--beta1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--beta2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
    parser.add_argument("--n_gpu", type=int, default=1, help="number of gpu to use during training model")
    parser.add_argument("--eval_interval", type=int, default=10, help="interval between model evaluating")
    parser.add_argument("--save_interval", type=int, default=10, help="interval between model saving")
    opt = parser.parse_args()
    print(opt)

    os.makedirs(opt.outputs_dir, exist_ok=True)
    os.makedirs(opt.ckpts_dir, exist_ok=True)

    device = torch.device("cuda:0" if (torch.cuda.is_available() and opt.n_gpu > 0) else "cpu")
    model = RetinexNet(device=device)

    train(opt, model=model)
    # evaluate(opt, model=model)
    # predict(opt, model=model)

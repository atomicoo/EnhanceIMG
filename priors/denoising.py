# %%
import torch
import torch.nn as nn
import torch.optim as optim
from skimage.metrics import peak_signal_noise_ratio
# from skimage.metrics import structural_similarity

try:
    from .datasets import *
    from .networks import *
except ImportError:
    from datasets import *
    from networks import *

# %%
device = torch.device('cpu')

# %%
imsize = -1
sigma = 25
sigma_ = sigma/255.
reg_noise_std = 1./30.  # set to 1./20. for sigma=50
channels = 3
figsize = 5
lr = 0.01
show_freq = 300
exp_weight = 0.99
num_iter = 4805

# %%
# fpath = "../Madison.png"
# img_pil, img_np = load_image(fpath, imsize=imsize)
# # img_np = random_crop(img_np, (384, 512))
# # img_np = center_crop(img_np, (384, 512))
# img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)

# %%
fpath = "./data/snail.jpg"
img_noisy_pil, img_noisy_np = load_image(fpath, imsize=imsize)
# img_noisy_np = random_crop(img_noisy_np, (256, 384))
img_noisy_np = center_crop(img_noisy_np, (256, 384))
img_np = img_noisy_np  # no ground-truth image of snail.jpg

# %%
grid = plot_image_grid([img_np, img_noisy_np], nrow=2, factor=0)

# %%
net = SkipNet(3, 3, nf=8, nf_ratios=[1,2,4,8,8], skip_channels=[0,0,0,4,4], skip_mode='skip', 
              upsample_mode='bilinear').to(device)
# Compute number of parameters
s = sum([np.prod(list(p.size())) for p in net.parameters()]); 
print ('Number of params: %d' % s)

# %%
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr)

# %%
input = get_noise(channels, 'noise', img_noisy_np.shape[1:]).to(device)
img_noisy_torch = np_to_torch(img_noisy_np).to(device)

# %%
input_cached = input.detach().clone()
noise = input.detach().clone()
output_avg = None
last_net = None
psrn_noisy_last = 0.

# %%
for i in range(num_iter):

    if reg_noise_std > 0:
        net_input = input_cached + (noise.normal_() * reg_noise_std)

    optimizer.zero_grad()

    output = net(input)

    if output_avg is None:
        output_avg = output.detach()
    else:
        output_avg = output_avg * exp_weight + output.detach() * (1 - exp_weight)

    loss = criterion(output, img_noisy_torch)
    loss.backward()

    optimizer.step()

    psrn_noisy = peak_signal_noise_ratio(img_noisy_np, output.detach().cpu().numpy()[0])
    psrn_gt    = peak_signal_noise_ratio(img_np, output.detach().cpu().numpy()[0])
    psrn_gt_sm = peak_signal_noise_ratio(img_np, output_avg.detach().cpu().numpy()[0])
    
    print(f"[Iter {i:05d}] Loss {loss.item():.4f} | "
           "PSNR_NOISY: {psrn_noisy:.4f} | PSRN_GT: {psrn_gt:.4f} | PSNR_GT_SM: {psrn_gt_sm:.4f}", '\r', end='')

    if i % show_freq == 0:
        output_np = torch_to_np(output)
        output_avg_np = torch_to_np(output_avg)
        plot_image_grid([np.clip(output_np, 0, 1), np.clip(output_avg_np, 0, 1)], factor=figsize, nrow=2)

        if psrn_noisy - psrn_noisy_last < -5: 
            print('Falling back to previous checkpoint.')
            for new_param, net_param in zip(last_net, net.parameters()):
                net_param.data.copy_(new_param.to(device))
        else:
            last_net = [param.detach().cpu() for param in net.parameters()]
            psrn_noisy_last = psrn_noisy

        cat_img = np.concatenate([img_noisy_np, output_np, output_avg_np], axis=2)
        cat_img = np_to_pil(cat_img)
        cat_img.save(f"demo{i:04d}.png")

# %%

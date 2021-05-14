# EnhanceIMG

[TOC]

此代码库用于图像增强算法的探索，主要包括：低光增强、图像修复、超分辨率重建 …… 

## 目录结构

```
.
|--- asserts/
|--- awegan/      # GAN相关算法
     |--- datasets/
     |--- models/         # CycleGAN/Pix2Pix/SelfGAN
     |--- options/
     |--- util/
     |--- __init__.py
     |--- train.py
     |--- ...
|--- colorspace/  # 色彩空间转换
|--- edges/       # 边缘检测算法
|--- filters/     # 各种滤波器
|--- histeq/      # 直方图均衡算法
|--- noises/      # 噪声
|--- priors/      # 自然图像先验信息
     |--- __init__.py
     |--- denoising.py
     |--- inpainting.py
     |--- networks.py     # ResNet/SkipNet/UNet
     |--- restoration.py
     |--- ...
|--- retinex/     # Retinex系列算法
     |--- __init__.py
     |--- enhancer.py
     |--- retinex_net.py  # RetinexNet
     |--- ...
|--- utils/       # 一些方法
|--- .gitignore
|--- demo.py
|--- LICENSE
|--- Madison.png
|--- README.md    # 说明文档
|--- requirements.txt     # 依赖文件
```

## 简单示例

### 添加噪声

**噪声**（原图|椒盐噪声|高斯噪声）

![noises](https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/04/1619614042-noises.png)

### 各种滤波器

**滤波器**（椒盐噪声|均值滤波|中值滤波）

![filters1](https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/04/1619614242-filters1.png)

**滤波器**（高斯噪声|高斯滤波|双边滤波|联合双边滤波）

![filters2](https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/04/1619614258-filters2.png)

**滤波器**（高斯噪声|引导滤波）

![filters3](https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/04/1619614271-filters3.png)

### 边缘检测

**检测算子**（灰度图|Laplacian|Sobel|Scharr）

![opt-edge-detection-2](https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/05/1619852372-opt-edge-detection-2.png)

**检测算子**（灰度图|LoG|DoG|Gabor）

![opt-edge-detection-3](https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/05/1620812279-opt-edge-detection-3.png)

**其他算法**（灰度图|结构森林|HED|HED-feats-5）

![hn-edge-detection](https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/05/1619852478-hn-edge-detection.png)

![hed-fs1-fs5](https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/05/1619955819-hed-fs1-fs5.png)

### 传统增强算法

**直方图均衡**（原图|HE|AHE|CLAHE）

![hist-equal](https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/04/1619614292-hist-equal.png)

**Gamma 校正**（原图|Gamma|Gamma+MSS）

![adjust-gamma](https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/04/1619684267-adjust-gamma.png)

**Retinex**（原图|MSRCR|AMSRCR|MSRCP）

![retinex](https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/04/1619614304-retinex.png)

**Retinex 增强**（原图|AttnMSR|AttnMSR+MSS）（Mine）

![enlighten](https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/04/1619614316-enlighten.png)

### 神经网络

**RetinexNet**（原图|RetinexNet）

![retinexnet](https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/04/1619668202-retinexnet.png)

### 生成对抗网络

**Pix2Pix**

（边缘 <=> 图像）

![pix2pix-facades](https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/05/1620004141-pix2pix-facades.png)

（低光 <=> 正常）

![pix2pix](https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/05/1619955841-pix2pix.png)

![pix2pix4](https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/05/1620038713-pix2pix4.png)

**CycleGAN**

（夏天 <=> 冬天）

![summer2winter](https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/05/1619937669-summer2winter.png)

（低光 <=> 正常）

![cyclegan4](https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/05/1620037334-cyclegan4.png)

## 参考资料

- [Multiscale Retinex](http://www.ipol.im/pub/art/2014/107/)
- [An automated multi Scale Retinex with Color Restoration for image enhancement](http://ieeexplore.ieee.org/document/6176791/)
- [A multiscale retinex for bridging the gap between color images and the human observation of scenes](http://ieeexplore.ieee.org/document/597272/)
- [Deep Retinex Decomposition for Low-Light Enhancement](https://arxiv.org/abs/1808.04560)
- [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)
- [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
- [High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs](https://arxiv.org/abs/1711.11585)
- [Toward Multimodal Image-to-Image Translation](https://arxiv.org/abs/1711.11586)
- [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1605.06211v1)

## TODO

- [x] AttnMSR 图像增强算法（Mine）
- [x] RetinexNet 低光增强模型
- [x] ResNet / SkipNet / UNet
- [ ] Deep Image Prior（自然图像先验信息）
- [x] Pix2Pix 模型用于图像增强
- [x] CycleGan 模型用于图像增强
- [ ] SelfGAN 图像增强模型（Mine，完善中）

## 欢迎交流

- 微信号：Joee1995

- 企鹅号：793071559


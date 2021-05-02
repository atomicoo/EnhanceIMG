# EnhanceIMG

[TOC]

此代码库用于图像增强算法的探索。

## 目录结构

```
.
|--- asserts/
|--- awegan/      # GAN相关算法
     |--- __init__.py
     |--- cyclegan.py
     |--- dcgan.py
     |--- selfgan.py
     |--- ...
|--- colorspace/  # 色彩空间转换
|--- edges/       # 边缘检测算子
|--- filters/     # 各种滤波器
|--- histeq/      # 直方图均衡算法
|--- noises/      # 噪声
|--- retinex/     # Retinex系列算法
     |--- __init__.py
     |--- enhancer.py
     |--- retinex_net.py
     |--- ...
|--- utils/       # 一些方法
|--- .gitignore
|--- demo.py
|--- LICENSE
|--- Madison.png
|--- README.md    # 说明文档
|--- requirements.txt  # 依赖文件
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

**其他算法**（灰度图|结构森林|HED|HED-feats-5）

![hn-edge-detection](https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/05/1619852478-hn-edge-detection.png)

![hed-fs1-fs5](D:\Code\bishe\EnhanceIMG\asserts\hed-fs1-fs5.png)

### 传统增强算法

**直方图均衡**（原图|HE|AHE|CLAHE）

![hist-equal](https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/04/1619614292-hist-equal.png)

**Gamma 校正**（原图|Gamma|Gamma+MSS）

![adjust-gamma](https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/04/1619684267-adjust-gamma.png)

**Retinex**（原图|MSRCR|AMSRCR|MSRCP）

![retinex](https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/04/1619614304-retinex.png)

**Retinex 增强**（原图|AttnMSR|AttnMSR+MSS）

![enlighten](https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/04/1619614316-enlighten.png)

### 神经网络

**RetinexNet**（原图|RetinexNet）

![retinexnet](https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/04/1619668202-retinexnet.png)

### 生成对抗网络

**Pix2Pix**

（边缘 <=> 图像）TODO：待补图

（低光 <=> 正常）

![pix2pix](D:\Code\bishe\EnhanceIMG\asserts\pix2pix.png)

**CycleGAN**

（夏天 <=> 冬天）

![summer2winter](https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/05/1619937669-summer2winter.png)

（低光 <=> 正常）TODO：待补图

## TODO

- [x] RetinexNet 低光增强模型
- [x] Pix2Pix 模型用于图像增强
- [ ] CycleGan 模型用于图像增强
- [ ] SelfGAN 图像增强模型（Ours）

## 欢迎交流

- 微信号：Joee1995

- 企鹅号：793071559
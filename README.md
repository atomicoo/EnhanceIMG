# EnhanceIMG

[TOC]

此代码库用于图像增强算法的探索。

## 目录结构

```
.
|--- awegan/      # GAN相关算法
|--- colorspace/  # 色彩空间转换
|--- filters/     # 各种滤波器
|--- histeq/      # 直方图均衡算法
|--- noises/      # 噪声
|--- retinex/     # Retinex系列算法
|--- utils/       # 一些方法
|--- demo.py
|--- LICENSE
|--- Madison.png
|--- README.md    # 说明文档
|--- requirements.txt  # 依赖文件
```

## 简单示例

### 添加噪声

**噪声**（原图|椒盐噪声|高斯噪声）

![noises](https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/04/noises.png)

### 各种滤波器

**滤波器**（椒盐噪声|均值滤波|中值滤波）

![filters1](https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/04/filters1.png)

**滤波器**（高斯噪声|高斯滤波|双边滤波|联合双边滤波）

![filters2](https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/04/filters2.png)

**滤波器**（高斯噪声|引导滤波）

![filters3](https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/04/filters3.png)

### 传统增强算法

**直方图均衡**（原图|HE|AHE|CLAHE）

![hist-equal](https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/04/hist-equal.png)

**Retinex**（原图|MSRCR|AMSRCR|MSRCP）

![retinex](https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/04/retinex.png)

**Retinex 增强**（原图|AttnMSR|AttnMSR+MSS）

![enlighten](https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/04/enlighten.png)

### 生成对抗网络

**风格迁移**（夏冬转换）

![summer2winter](https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/04/summer2winter.png)


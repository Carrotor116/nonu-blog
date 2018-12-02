---
title: 'Feature Super-Resolution: Make Machine See More Clearly'
id: feature-super-resolution
category: paper
date: 2018-12-02 13:32:13
tags: ['super resolution', 'GAN']
mathjax: true
---

本文提出 Feature Super-Resolution 概念，基于 GAN 实现了图像特征的超像素还原。提高了机器对低分辨率图像的鉴别能力，能有效改善基于低分辨率的计算机应用，如图像检索。

<!-- more -->

>  论文：[Feature Super-Resolution: Make Machine See More Clearly](http://openaccess.thecvf.com/content_cvpr_2018/html/Tan_Feature_Super-Resolution_Make_CVPR_2018_paper.html)



---



## Intro

### Effect of Down-scaling Operation

使用 VGG16 模型和 Oxford5K 数据集来评估缩小操作对 **深层表示** (the deep representations) 的影响。

通过计算高分辨率和低分辨率图像的深层特征 (deep features) 之间的欧式距离均值，来衡量影响，有以下结果

* 随着缩小率的增大，其深层特征之间的距离均值变大

使用低分辨率图像进行图像检索实验，评估使用速度特征时低分辨率对匹配、检索的影响，得到结果

* 随着缩小率的增大，检索精度 (mAP) 急剧下降

为了了解不同分辨率图像深层特征之间的关系，计算同一缩小率下欧式距离的方差，得到结果

* 不同缩小率内的欧式距离方差的相近的。推出结论，深度特征的变化依赖于丢失的信息，而非特定的图像内容

基于以上观察，提出 Feature Super-Resolution Generative Adversarial Network ( FSR-GAN ) 模型，以提高机器对该图像的鉴别鉴别力。

> 以低分辨率图像特征为基础，生成还原高分辨率图像的特征。而非在低分辨率图像特征基础上进一步提取深层特征。因为是一个对特征的“还原操作”，所以类似于图像超分辨



## The Method (FSR-GAN)

FSR-GAN 包含两个子网络，特征生产网络 $G$ 和 特征鉴别网络 $D$ 。

$G$ 为 CNN，输入低分辨率图像特征 $F^{LR}$，输出超分辨率图像特征 $F^{SL}$ ，学习的是低分辨率图像特征 $F^{LR}$ 和高分辨率图像特征 $F^{HR}$ 之间的关系。

$D$ 也是一个 CNN，输入图像特征，判断该特征属于 超分辨率特征 $F^{SR}$ 还是 高分辨率特征 $F^{HR}$。

与传统 GAN 不同的时，本文提出了一个 focal loss 用于**强化对高缩小率样本的学习**。

{% asset_img fsr-gan.png %}


本文使用下采样生产低分辨率图像，使用 VGG16 模型进行原始图像特征的提取，可表示为
$$
\begin{aligned}& F^{LR} = F(I^{LR})\\
& F^{HR} = F(I^{HR})\end{aligned}
$$

### Focal Loss Function

原始的 GAN 网络中生产网络和鉴别网络的损失函数为
$$
\begin{aligned}
& L(G)=E_{x\sim P_g}[1-\log D(x)] \\
& L(D)=-E_{x\sim P_r}[\log D(x)] - E_{x\sim P_g}[1-\log D(x)]
\end{aligned}
$$
其缺陷在于分类器效果越好的时候，生成器会出现严重的梯度消失。WGAN 对此进行了改进，提出了以下损失函数
$$
\begin{aligned}
& L(G)=-E_{x\sim P_g}[D(x)]\\
& L(D)=E_{x\sim P_g}[D(x)]-E_{x\sim P_r}[D(x)]
\end{aligned}
$$
本文中直接使用 WGAN 不能得到良好的效果，进而使用均方误差 MSE 强化对生成网络的约束条件，得到如下生成网络损失函数
$$
L(G)=-E_{x\sim P_g}[D(x)]+\frac{1}{m}\sum_{i=1}^{m}(||F_i^{SR} -F_i^{HR}||_2)
$$
上式损失函数未考虑样本的不均衡性，即高缩小率原本对网络的影响应该比低缩小率样本的高。鉴于论文 14 [^14] 的 focal cross entropy loss ，得到 focal loss
$$
L(G)=-E_{x\sim P_g}[D(x)]+\frac{1}{m}\sum_{i=1}^{m}(||F_i^{SR} -F_i^{HR}||_2)^r
$$
其中 $r$ 为 focal loss 的权重。实验显示，$r$ 取值 2 的时候，$F^{SR}$ 和 $F^{HR}$ 之间的距离最小。



### Implementations

特征生成网络：

| type         | kernel  size | stride | channel | output  size |
| ------------ | ------------ | ------ | ------- | ------------ |
| convolution  | 8 × 8        | 1      | 4       | 64 × 64 × 4  |
| convolution  | 5 × 5        | 2      | 8       | 32 × 32 × 8  |
| convolution  | 5 × 5        | 1      | 16      | 32 × 32 × 16 |
| convolution  | 5 × 5        | 2      | 32      | 16 × 16 × 32 |
| convolution  | 5 × 5        | 1      | 64      | 16 × 16 × 64 |
| convolution  | 5 × 5        | 2      | 128     | 8 × 8 × 128  |
| dropout(70%) |              |        |         | 1 × 64 × 128 |
| linear       |              |        |         | 1 × 4096     |

特征鉴别网络：


| type        | kernel size | stride | channel | output size  |
| ----------- | ----------- | ------ | ------- | ------------ |
| convolution | 5 × 5       | 2      | 8       | 32 × 32 × 8  |
| convolution | 5 × 5       | 2      | 16      | 16 × 16 × 16 |
| convolution | 3 × 3       | 2      | 32      | 8 × 8 × 32   |
| convolution | 3 × 3       | 1      | 64      | 8 × 8 × 64   |
| linear      |             |        |         | 1            |

两个模型中各层均采用 Leaky ReLU 作为激活函数。损失函数使用 Adam 算法优化，学习率 0.0008，focal loss 权重 $r$ 取 2 ，epoch 取 6，采用 tensorflow 框架实现。



## Experimental

对比方法：ISR 方法 SRCNN 、VDSR

数据集：Oxford5K（4500训练、562评估）、 INRIA Holidays（500评估，其余的训练）、和 Paris datasets （612评估，其余的训练）。下采样率 1/4 、1/9 和 1/16

指标：欧式距离均值

结果：FSR-GAN 显著减少了低分辨率和高分辨率的表示 (the representation)，且在各缩小率之间的距离比较稳定，也比相似图像和高分辨率图像的距离要小。ISR 算法只在较低缩小率下的表现比 FSR-GAN 好。



## Applications

通过进行图像检索应用，测试 FSR-GAN 的强化特征表现结果。

进行了 Content Based Image Retrieval、Large-Scale Image Retrieval 和 Low Bit-Rate Mobile Visual Search应用上的测试，查询图像为下采样后再插值成 224 x 224 ( VGG16 的要求输入尺寸) 的低分辨率图像。数据集为Oxford5K 、Paris 和 INRIA Holidays。评价指标为 mean Average Precision。



**Content Based Image Retrieval 和 Large-Scale Image Retrieval ：**

结果为 FSR-GAN 能有效提高 mAP，且在各缩小率下均有相对稳定的改善。而 ISR 方法的改善效果不明显。

在 Holidays 数据集上，低分辨率的检索甚至比原始分辨率的 mAP 有所提高。论文解释为该数据集上与每个查询关联的图像数量较少，容易导致检索精度的波动。

> This phenomenon is caused by the characteristic of Holidays dataset. In Holidays dataset, the number of images associated with each query is small (about 4 images), which easily results in the fluctuation of retrieval accuracy



**Low Bit-Rate Mobile Visual Search ：**

在 移动设备视觉检索时，由于无线网络特点，该类型检索的响应时间与传输信息量有很大关系 (传输时延)。因此减少传输信息对此很重要，所以通过下采用图像，以减少客户端与服务器之间的数据传输，并且服务端使用 FSR-GAN 还原图像特征进行图像检索。实验结果显示在不同比他率下，FSR-GAN 都显著改善了检索结果。





## Annotation

focal loss: [Focal Loss - AI之路 - CSDN博客](https://blog.csdn.net/u014380165/article/details/77019084)

[^14]: [Object retrieval with large vocabularies and fast spatial matching - IEEE Conference Publication](https://ieeexplore.ieee.org/abstract/document/4270197)

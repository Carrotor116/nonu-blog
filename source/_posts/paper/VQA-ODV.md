---
id: VQA-ODV
title: VQA-ODV
category: paper
date: 2019-01-23 08:42:42
mathjax: true
tags:
---


本文作者给出了具有主观分数以及头部运动和眼部运动数据的全景视频 VQA 数据集。并且提出了基于深度学习并且整合了头部运动和眼部运动数据的全景视频 VQA 方法。

<!--more-->

> 论文: [Bridge the Gap Between VQA and Human Behavior on Omnidirectional Video: A Large-Scale Dataset and a Deep Learning Model](https://arxiv.org/abs/1807.10990)

---



## VQA-ODV

### Database

* reference
  ```yml
  video_num: 60         # 10 groups，6 per group
  resolution: [4K, 8K]  # a range, 3840 × 1920 pixels to 7680 × 3840 pixels
  projection: 'ERP'
  duration: [10, 23]    # a range, unit: seconds
  frame: [24, 30]       # a range, unit: frames per seconds (fps)
  ```

  {% asset_img reference.png %}


* impaired

  ```yaml
  video_num: 540            # 3 compress * 3 projection * 60 references
  compression_std: H.256
  QP: [27, 37, 42]      # quantization parameter, correspond to compress level
  projection: ['ERP', 'RCMP', 'TSP']
  ```

  {% asset_img impaired.png %}



### Subjective data formats

数据集给出了 `MOS` 和 `DMOS` ，其中 `MOS` 的计算如下
$$
{\rm MOS}_j = \frac{1}{I_j}\sum_{i=1}^{I_j}S_{ij}
$$
其中 $S\_{ij}$ 表示第 $i$ 个人对序列 $j$ 的评分，$I\_j$ 表示第 $j$ 个序列的有效评分人数。

`DMOS` 计算参考论文 [^29] 

### Content and formats

在一个序列上，一个用户的 `HM` 和 `EM` 表示如下

`[Timestamp HM_pitch HM_yaw HM_roll EM_x EM_y EM_flag]`

* Timestamp: 两个相邻采样点的时间间隔，单位毫秒
* HM data: 使用三个维度的欧拉角表示，pitch / yaw / roll 
* EM data: 使用 x / y 表示 viewport 中的位置，x / y 属于 `[0,1]`
* EM flag: 标记 RM 是否有效，1 为有效，0为无效。因为眨眼会使 `EM` 无效，而 `HM` 则是一直有效的。

### Analysis on VQA Results

主观质量 `DMOS` 分析

1. 相同投影方式下，高比特率的视频序列具有更好的主观质量
2. 相同比特率下，TSP 投影的主观质量比其他两类好
3. 高比特率下，投影方式带来的主观质量差异不显著

客观分数 (`PSNR`/`S-PSNR`/`SSIM`) 分析其与对应序列的 `DMOS` 的拟合相关性 (`SRCC`/`PCC`/`RMSE`/`MAE`)

1. `S-PSNR` 的相关性比传统的 `PSNR` 和 `SSIM` 指数要好

### Analysis on Human Behavior

1. 不同人的 `HM` 权重图之间存在高的一致性
2. 不同人的 `EM` 权值图也存在高的一致性
3. 所有人的 viewport region 相对于整个全景视频区域的比例小于 65%，即有感知冗余

### Impact of Human Behavior on VQA

在计算有损序列的帧的 PSNR 时引入感知权重，即 `HM`/`EM` 数据。

考虑三类感知权重，overall HM (O-HM) / individual HM (I-HM) / individual EM (I-EM) 。记用户 $i$ 的 viewport 对应的像素集合为 $\mathbb{V}\_i$ 

用户 $i$ 对一全景帧的 `I-HM` 权重图 $w\_i^{I-HM}$ 
$$
w_i^{I-HM}({\rm p}) = \begin{cases}1, {\rm p}\in \mathbb{V}_i \\ 0, {\rm p}\in {\rm others} \end{cases}
$$
${\rm p}$ 表示像素。

每个全景帧的 `O-HM` ，为加权每个用户的 `I-HM`
$$
w^{O-HM}({\rm p}) =\frac{\sum_{i=1}^I w_i^{I-HM}({\rm p})}{\sum_{\rm p \in \mathbb{P} }\sum_{i=1}^I w_i^{I-HM}({\rm p})}
$$
$\mathbb{P}$ 表示全景帧的所有像素。

`I-EM` 权重图 $w\_i^{I-EM}$ 是高斯形式
$$
w_i^{I-EM}({\rm p}) = \begin{cases}
\exp \left(-\frac{||{\rm e_p} - {\rm e}_i||_2^2}{2\sigma^2}\right),{\rm p}\in \mathbb{V}_i\\
0, {\rm p} \in {\rm others}
\end{cases}
$$
${\rm e}\_i$ 是用户 $i$ 的 `EM` 位置，${\rm e\_p}$ 是 viewport 中像素 ${\rm p}$ 的位置。

然后对失真序列和参考序列之间的误差加权，计算 ${\rm PSNR\_{I-HM} }$ / ${\rm PSNR\_{O-HM} }$ / ${\rm PSNR\_{I-EM} }$ 
$$
{\rm PSNR_{I-EM} } = \frac{1}{I}\sum_{i=1}^I10\log\frac{Y^2_{\max}\cdot\sum_{p\in\mathbb{P} }w_i^{I-EM}({\rm p})}{\sum_{ {\rm p}\in\mathbb{P} }(Y({\rm p}) -Y'({\rm p}))^2\cdot w_i^{I-EM}({\rm p})}
$$

$$
{\rm PSNR_{I-HM} } = \frac{1}{I}\sum_{s=1}^I10\log\frac{Y^2_{\max}\cdot\sum_{p\in\mathbb{P} }w_i^{I-HM}({\rm p})}{\sum_{ {\rm p}\in\mathbb{P} }(Y({\rm p}) -Y'({\rm p}))^2\cdot w_i^{I-HM}({\rm p})}
$$

$$
{\rm PSNR_{O-HM} } =10\log\frac{Y^2_{\max} }{\sum_{ {\rm p}\in \mathbb{P} }(Y({\rm p})-Y'({\rm p}))^2\cdot w^{O-HM}({\rm p})}
$$

在计算以上 ${\rm PSNR}$ 和 `DMOS` 之间的相关度，得出 `HM` 和 `EM` 能提高客观质量分数的结论。



## Deep Learning Base VQA Model

{% asset_img deep_model.png %}


输入失真序列和参考序列。

预处理过程，计算失真序列和参考序列的误差，对给定 patch 对应的 `HM` 和 `EM` 权值分别求和作为 path 的权值，然后对每个全景序列采样 n 个失真 patches (size: 112 x 112)，采样的概率是 patches 的 `HM` 权值。同样计算 n 个失真 patches 的误差图。

接下来将失真 patch 和对应的误差图输入<u>一个计算二维图像的 VQA 的卷积神经网络组件（ DeepQA 组件）</u> [^13] ，得到 n 个局部 VQA 分数。将 n 个局部分数拼接成 $n\times 1$ 的向量。同时将 n 个失真 patches 的权重除这些权值和得到一个 n 维的权重向量。最后接两个全连接层输出 Objective VQA score。

损失函数含有三项
$$
\begin{aligned}
\zeta =& \lambda_1 \underbrace{||s-s_g||_2^2 }_{\rm Mean\quad square\quad error}\\
& + \lambda_2\underbrace{\frac{1}{nHW}\sum_{k=1}^n\sum_{(h,v)}({\rm Sobel}_h({\rm M}_k)^2 + {\rm Sobel}_v({\rm M}_k)^2)^{\frac{3}{2} }}_{\rm Total\quad variation\quad regularization} \\
& +\lambda_3\underbrace{||\beta||_2^2}_{\rm L_2\quad regularization}
\end{aligned}
$$
其中 ${\rm s}$ 和 ${\rm s\_g}$ 分别为预测客观 VQA 分数向量和 ground truth `DMOS` 分数。${\rm M}\_k$ 是 $k$-th CNN 组件输出的 sensitivity map。${\rm Sobel}\_h$ 和 ${\rm Sobel}\_v$ 分别是在像素坐标系中水平和垂直的 ${\rm Sobel}$ 操作。$\beta$ 是网络的所有训练参数，$\lambda\_i​$ 是权值。

* Mean square error: 测量客观分数和主观分数之间的欧式距离
* Total variation regularization: <u>惩罚高频分量内容，因为人眼对高频分量不敏感</u>
* ${\rm L}\_2$ regularization: 防止过拟合

### Evaluation 

随机取 12 个参考序列及其 108 个失真序列作为测试集，其余 48 个参考序列和 432 个失真序列作为训练集。并且将序列进行空间下采样到 960 像素，时间下采样到 45 帧。<u>并且使用预测的 `HM` 图和 `EM` 图</u>。

对比对象：DeepQA 、S-PSNR 、CPP-PSNR 、WS-PNSR （第一属于深度学习方法）

对比方式：客观分数与主观分数之间的相关度（ PCC 、SRCC 、RMSE 和 MAE ）

结果：使用预测 `HM` 图和 `EM` 图时，本文方法显著优于其他方法。

消融实验：使用 group truth `HM` 图和 `EM` 图，本文方法有进一步提升，表示 `HM` 图和 `EM` 图的精度会影响本文方法。



## Problem

1. What is the high frequency detail ?
> Total variation (TV) regularization. It is applied to penalize the high frequency content as an smoothing constraint, since human eyes are insensitive to high frequency detail
> which has the subjective scores of 600 omnidirectional sequences

2. What is the EM prediction in the model ? How to calculate it ?
> This indicates that the accuracy of HM and EM prediction influences the performance of our VQA approach.



## Annotation 

[^13]: [Deep Learning of Human Visual Sensitivity in Image Quality Assessment Framework - IEEE Conference Publication](https://ieeexplore.ieee.org/abstract/document/8099696)
[^29]: [Study of Subjective and Objective Quality Assessment of Video - IEEE Journals & Magazine](https://ieeexplore.ieee.org/abstract/document/5404314)
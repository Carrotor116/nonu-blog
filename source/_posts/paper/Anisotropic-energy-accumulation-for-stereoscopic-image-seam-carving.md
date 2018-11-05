---
title: Anisotropic energy accumulation for stereoscopic image seam carving
id: aseec
date: 2018-09-29 12:41:58
tags: ['Stereo image resizing', 'stereo seam carving', 'accumulative energy', 'forward energy', '3D structure energy']
mathjax: true
---

接缝裁剪是一种特征图大小以适应显示的技术，通过填充或删除接缝实现。重复的删除或插入接缝会使得图片失真明显，为解决视觉失真，本文提出了一种新的 AESSC 方法。

<!-- more -->

1. 在删除和插入seams时，将seams 上像素的能量向其周围像素分散
2. 使用Sobel operator 检测具有最大边缘信息的方向，并且沿着该方向继续能量积累
3. 在强度函数上融入3D结构一致性限制，并且采用像素视觉性维护方法

实验表明，该方法能有效减少在维护**立体图像**的几何一致性时产生的视觉失真

> 论文：[Anisotropic energy accumulation for stereoscopic image seam carving](https://ieeexplore.ieee.org/document/8251899)



---



## AESSC

### 定义 

`seams` [^20] ：

1. 输入$m\times n$ 的立体图像
2. 使用立体匹配算法(stereo matching algorithm)，如SGM 计算视差图（disparity map）
3. 通过视差图计算 seam 的几何耦合 $S\_{L}=\\{s\_{l}^{i}\\}\_{i-1}^{m}$ 和 $S\_{R}=\\{s\_{r}^{i}\\}\_{i-1}^{m}$

左视图中 seam 的第 $i$ 行像素应满足 $s\_{l}^{i}=(i, j\_{L}(i)) \in S\_{L}$， 如果对应右视图中的像素是 $s\_{r}^{i}=(i,j\_{R}(i))\in S\_{R}$， 则将有如下关系
$$
\begin{equation*}
\begin{aligned}
& s_{R}^{i} = (i,j_{R}(i)) = (i, j_{L}(i)+D(S_{L}^{i}))\\
& s_{R}^{i} = (i,j_{R}(i)) \in S_{R}\\
& s_{L}^{i} = (i,j_{L}(i)) \in S_{L}
\end{aligned}
\end{equation*}
$$
其中 $j\_{L},j\_{r}:[m] \to [n]$

`强度项`(the intensity term) [^20] ：用于最小化每个（左右）视图的失真。通过测量左右视图中的结果梯度进行，该左右视图使用移除和插入seams 像素调整了大小

`3D几何项`(the 3D geometry term) [^20] ：用于最小化视差失真（disparity distortion）。通过测量视差图中的结果梯度进行

### 基于能量累积的动态规划

将seam的像素能量分该像素的邻居像素，按四个方向，水平、垂直、45° 和 135°

{% asset_img fig_1.png "An example of accumulated energy update. (a) and (b) are the energy matrix before and after removing green pixel, respectively. (c) shows the way to continue the accumulation of energy in the horizontal direction" %}


权重计算
$$
\begin{equation*}
\begin{aligned}
& w_{1} = max(M(i, j-1), M(i, j+1))\\
& w_{2} = max(M(i-1, j), M(i+1, j))\\
& w_{3} = max(M(i-1, j-1), M(i+1, j+1))\\
& w_{4} = max(M(i-1, j+1), M(i+1, j-1))\\
& w_{i} = w_{i} / (2\times \sum(w_{i}+w_{c}+\sigma)), \quad i=1,2,3,4
\end{aligned}
\end{equation*}
$$
其中最后一个式子为归一化，$w\_{c}$ 是分配给另外两个像素的能量权值，$\sigma$ 是防止分母为0，取值 0.0001

能量分配
$$
\begin{equation*}
\begin{aligned}
& M(i,j\pm 1) =M(i,j\pm 1) +w_{1}\times M(i,j)\\
& M(i\pm 1,j ) =M(i\pm 1,j) +w_{2}\times M(i,j)\\
& M(i\pm 1,j\pm 1) =M(i\pm 1,j\pm 1) +w_{3}\times M(i,j)\\
& M(i\pm 1,j\mp 1) =M(i\pm 1,j\mp 1) +w_{4}\times M(i,j)
\end{aligned}
\end{equation*}
$$


### 能量累积的边检测

使用 Sobel edge detection operator 计算像素四个方向的边强度（the strength of edge），其模板为
$$
\begin{equation*}
\begin{aligned}
& a0° = \begin{bmatrix}
1&2&1\\
0&0&0\\
-1&-2&-1
\end{bmatrix}\quad
a90° = \begin{bmatrix}
1&0&-1\\
2&0&-2\\
1&0&-1
\end{bmatrix}\\
& a45° = \begin{bmatrix}
2&1&0\\
1&0&-1\\
0&-1&-2
\end{bmatrix}\quad
a135° = \begin{bmatrix}
0&-1&-2\\
1&0&-1\\
2&1&0
\end{bmatrix}
\end{aligned}
\end{equation*}
$$
根据边检测结果，在具有像素最大边能量的方向上，继续进行能量累积，如
$$
\begin{equation*}
M(i, j\pm 2) =M(i, j\pm 2)  + w_{1}/2\times M(i,j)
\end{equation*}
$$
其中 $w\_{c} = w\_{1} /2$ 作为权，与最大边能量的方向有关



## 实验 

数据集：NJUDS2000（含有2000张立体图像）

深度图 [^20] ： Sun's optical flow 方法计算得到。（表示能量函数中的深度能量）

视差图：semiglobal matching (SGM) 方法计算得到

缩放范围：从 $h\times w$  到 $h\times 0.83w$

对比方法：SSC [^20] ，homogeneous scaling，OSS [^30]

结果：本文方法可以有效维持显著物体的形状和尺寸

## word

seam carving  - [接缝裁剪](https://zh.wikipedia.org/zh/%E6%8E%A5%E7%B8%AB%E8%A3%81%E5%89%AA)

disparity map  - 视差图

occlusion  - 闭包

geometric coupling  - 几何耦合



[^20]: [Stereo Seam Carving a Geometrically Consistent Approach](https://ieeexplore.ieee.org/document/6468042)

[^30]: [Optimized scale-and-stretch for image resizing](https://dl.acm.org/citation.cfm?doid=1457515.1409071) 
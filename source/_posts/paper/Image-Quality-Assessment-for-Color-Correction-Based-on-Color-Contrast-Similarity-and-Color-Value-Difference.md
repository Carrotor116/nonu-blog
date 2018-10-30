---
title: >-
  Image Quality Assessment for Color Correction Based on Color Contrast
  Similarity and Color Value Difference
id: IQA4color-correction
date: 2018-09-20 15:29:02
tags: ['image quality assessment', 'color correction', 'visual saliency', 'color contrast similarity', 'color value difference']
mathjax: true
---

一种基于**颜色对比度**和**色差**的图像增强结果的评估因子。
<!-- more -->


> 论文：[Image Quality Assessment for Color Correction Based on Color Contrast Similarity and Color Value Difference](https://ieeexplore.ieee.org/document/7763834/)



---



## 原理步骤概述

1. 采用 SIFT 算法生成 **matching map** 代替 reference map (文中仍然指定描述为matching map)
2. 采用 SSIM 算法计算matching map 的confidence map
3. 采用 GBVS算法计算target map 的saliency map
4. 结合confidence map 和saliency map 生成 **weight map**
5. 对matching map 的各个颜色通道（文中采用CMYK颜色空间）分别计算三个评估组件生成对应similarity map $D\_{i}$，三个评估组件包括 **对比度相似度**、**色差均值**、**色差跨度**
6. 对similarity map 应用 weight map 计算出评估组件均值 $e\_{i}$
7. 按评估组件对 $e\_{i}$ 计算均值得到三个评估组件的均值 $V\_{i},i\in\\{1,2,3\\}$ ，对 $V\_{i}$ 进行按权线性求和得到最终评价指标值

{% asset_img fig_1.png "Flow diagram" %}

## 细节详述

### 计算matching map

**由于 FR-IQA 指数算法要求输入的两张图片具有相同的scene，而实际上reference image仅仅拥有与target image相似的scene，所以需要进行转换**

采用 SIFT 算法[11] ，对 reference image 和 target map 进行配准（registrate），并且从reference image 中获取像素信息，从而生成matching map。当reference image中不存在target image的部分区域时，称`no-matching`，该部分在mask map中使用纯黑区域表示。

由此得到的matching map 将具有和 reference map 相同的颜色信息，和target map具有相同的场景结构信息（对比 mask map 后的场景结构信息）。

（类似从reference map中抠图，当reference map 的场景信息大于target map 时使用裁剪，反之小于时，使用黑色填补空缺场景部分，以此生成matching map）

{% asset_img fig_2.png "fig_2" %}

### 计算confidence map 和saliency map

由于matching map 的结构信息已经和target map达到**很一致**的程度，但仍然未必能**完全一致**，引入SSIM 算法计算matching map 和target map 的结构信息，以获得 confidence map。该confidence map 能提各个区域结构相似度的权重信息。（如mask map 中纯黑表示的`no-matching`部分将在confidence map 中得到0的权重值。）

使用 GBVS[29]算法对target map计算saliency map。应用像素的显著性信息能够更好的模拟 HVS 特征。

confidence map 和saliency map 结合得到weighting map



### 对比相似度评估组件计算 （color contrast similarty）

对比相似度反映两张灰阶图中的亮度差异信息。

1. 转换RGB颜色空间到 CMYK 颜色空间，根据四个颜色通道生成对应的四个CS图，

2. 对比matching map 和target map 四个对应通道CS图，计算**对比度相似度**，评估出**对比度相似图**

3. 将四个对比度相似图应用weight map 计算四个通道的评估值 $e\_{i}$ ，对 $e\_{i}$ 求均值得到最终结果 $V\_{1}$

#### 对比相似度计算

$$
\begin{aligned}
& \mu_{x} = \sum_{i=1}^{N}w_{i}\times x_{i} \\
& \sigma_{x} = (\sum_{i=1}^{N}(w_{i}\times(x_{i}- \mu_{x})^{2})^{1/2}
\end{aligned}
$$
$x$ 和 $y$ 各自为matching map 和target map 的对应区域的具有相同圆心的`圆形对称高斯窗口`。$w =\\{w\_{i}|i=1, ..N\\}$ 为 $11 \times 11$的`圆形对称高斯权重函数`，（由于是权重，所以$\sum\limits\_{i=1}^{N}w\_{i} = 1$ 。）$x\_{i}$ 是 $x$ 上像素 $i$ 的值，所以$\mu\_{x}$ 表示应用`高斯权重`的**像素值**。$\sigma\_{x}$ 表示高斯窗口下的像素的**标准差**
$$
c(x,y)=(2\times \sigma_{x} \times \sigma_{y}+c1)/(\sigma_{x}^{2}+\sigma_{y}^{2} + c1)
$$
$c(x, y)$ 通过 $\sigma\_{x}$ 和 $\sigma\_{x}$ 来计算对比相似度（原型为 $\frac{2\times \sigma\_{x} \times \sigma\_{y}}{\sigma\_{x}^{2}+\sigma\_{y}^{2} }$ ），其中提供引入 $c1$ 来解决分母可能为0的情况，在本文中取值 $c1 = 0.1$ 。故 $c(x, y) \in (0,1)$ 表示`高斯窗口`的对比相似度值，通过滑动`高斯窗口`遍历整张图，组成**对比度相似图**。由于采用CMYK颜色空间，所以一张matching map 将产生4个**对比图相似图**，一个通道一张图 。

#### CS评估值计算

应用weighting map 计算对比图相似图的加权平均值 $e\_{n}, n\in \\{1, 2, 3, 4\\}$
$$
\begin{aligned}
& e_{n} = \frac{\sum_{i}^{p} \sum_{j}^{q}(m(i,j)\times D_{n}(i,j))}{\sum_{i}^{p} \sum_{j}^{q}m(i,j)} \\
& m(i,j) = M_{c}(i,j)\times M_{s}(i,j)
\end{aligned}
$$


$p$ 和 $q$ 为图片宽高，$M\_{c}$ 和 $M\_{s}$ 分别为confidence map 和saliency map，$\times$ 为像素乘。则 $m(i,j)$ 为对应像素的权值，$e\_{n}$ 为**加权平均值**。
$$
\bar{V1} = \frac{\sum_{i=1}^{4}e_{n}}{4}
$$
$\bar{V1}$ 为最终的颜色对比度相似度（CS）评估组件的评估值。



### 色差均值评估组件计算（AVD）

在 CMYK  颜色空间内对4个通道分别进行计算
$$
d(x,y)= \frac{1}{c2 \times (\mu_{x}-\mu_{y})^{2} + 1}
$$
同CS计算过程，$x$ 和 $y$ 表示对应`圆形高斯窗口`，$\mu\_{x}$ 和 $\mu\_{y}$ 表示应用`高斯权重`的像素值。$c2$ 是与颜色空间相关的可调参数，对于 CMYK 颜色空间取 $c2=0.002$ ，常数1由于防止分母为0。

同样由 $d(x,y)$ 得到**色差均值图**，然后与[CS评估值计算](#CS评估值计算)相同，计算得到 $e\_{n}$ 和色差均值评估组件的评估值 $\bar{V2}$



### 色差跨度评估组件计算（SVD）

一些非常差颜色一致性的局部区域可能导致整个输入图像之间颜色一致性的显着下降。由此提出基于色差跨度的评估组件
$$
\begin{aligned}
& s(X,Y)=(\sum_{i=1}^{k}fa(|X-Y|,i) - \sum_{i=1}^{k}fi(|X-Y|,i)) /k\\
& k = max(100.0001 \times p \times q)
\end{aligned}
$$
$X$、$Y$ 为对比图，$p$ 和 $q$ 为宽高，$|X-Y|$ 表示色差图，$fa$ 和 $fi$ 分别是色差图中的第 $i$ 个最大和最小的像素值。而 $s(X,Y)$ 即表达两张图之间的色差跨度（value difference span）
$$
h(x,y) =1-|x-y|\times s(X,Y)
$$
其中 $x$ 和 $y$ 是两张图中相同位置的像素，由于颜色通道的像素值先被标准化到 $[0, 1]$，所以 $h(x,y)$ 的取值范围也为 $[0,1]$，值越大表示差异越小。

由 $h(x,y)$ 得到色**差跨度图**，然后与[CS评估值计算](#CS评估值计算)相同，计算得到 $e\_{n}$ 和色差跨度评估组件的评估值 $\bar{V3}$



### 最终评估结果

采用线性混合模型计算
$$
\begin{aligned}
& V = \alpha \times \bar{V1} + \beta \times  \bar{V2}  + \gamma \times  \bar{V2},\\
& s.t. \quad \alpha \geq 0, \beta \geq 0, \gamma \geq 0, \alpha+\beta+\gamma=1
\end{aligned}
$$

使用线性模型，其中 $\alpha$、$\beta$、$\gamma$  可调整参数，本文中由实验得各自取值为 $\alpha=0.4, \beta=0.2,\gamma=0.4$ 



## 实验

**数据集：**一个创建的图像颜色矫正数据集 ICCD，和四个公共数据集 CCID2014 [27], CID2013 [51], TID2013 [52], and CSIQ [41]。其中 ICCD 包含4个无损视频中获取18对图片。每对图片来自同一个视频的两个帧。颜色矫正采用6中，每种由三个矫正程度，所以有 324（ $18\times 6\times3$ ）张颜色矫正目标图，以及18张原图。

**颜色矫正算法：**GC [4], PRM[5], GCT [1], GCT-CCS [2], ICDT [3], 和 CHM [8]六种。（可生成 1944 （$324\times 6$）张颜色矫正图）

**评价标准：**相关性correlation（measure by PLCC），准确性accuracy（measure by RMSE），单调性monotonicity（measure by SRCC and KRCC）

**与本文所提评价因子进行对比的因子：**17 个 FR-IQA 因子和 RIQMC 因子、CSSS 因子以及 SSIMc因子

**实验方法：**将图片映射到CMYK颜色空间，分别对 2268（324 target images and 1944 result images）张图计算 20 个图片质量评估因子，然后根据 20 个评估因子得分和 MOS 分数计算 4 个评价标准。对于本文提出的评价算法，还分别对其三个评价组件（CS、AVD、SVD）和权重图的使用进行单独的实验以测试其对总体评价算法的有效性。并且分别用 reference map、matching map、idea reference map 三者与target map进行对本文中的评价算法，以验证使用 matching map 的有效性。

**实验结果：**验证了本文所提评估因子对比其他因子由更好的效果，且验证了CS、AVD、SVD评估组件的有效性、权重图使用的有效性。并且结果显示，本文评估因子使用matching map 所产生的结果甚至比其他 16 中使用了 idea reference map 评估因子的结果要更理想。

**不足与发展：**matching map 的应用消除了结构差异评估的影响，但是却使得评估不完整，即未能评估`no-matching`部分增强效果。本文评估因子的计算可以通过探索更好的matching map 来达到更好的效果。
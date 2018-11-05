---
title: >-
  Salient Object Segmentation Based on Superpixel and Background Connectivity
  Prior
id: SOS_based_on_superpixel_and_BCP
category: paper
date: 2018-11-05 17:03:41
mathjax: true
tags: ['superpixel', ' salient object segmentation', 'GrabCut']
---
本文根据超像素和背景连接先验，提出两阶段的 GrabCut 方法实现图像的显著性对象分割

<!-- more -->

步骤:

1. 使用线性迭代分簇算法提取图像的超像素
2. 基于超像素图像，使用背景连接先验，在相对于图像边界的颜色空间上表征每个超像素的空间布局
3. 根据显著性和背景连接值，标记4类 superpixel-level seeds，使用 seeds 进行 superpixel-level GrabCut 完成第一次图像分割
4. 在超像素水平的 GrabCut 的分割结果中裁剪一个矩形区域，进行像素级水平的 GrabCut 以生成最终分割结果


> 论文: [Salient Object Segmentation Based on Superpixel and Background Connectivity Prior](https://ieeexplore.ieee.org/document/8478410)



---



## The Method 

1. 使用 SLIC [^28] 提取超像素集

2. 基于显著性和背景连接先验来标记4类超像素水平的 seeds 得到 labeling result

3. 使用 labeling result 和 GrabCut 方法得到 `superpixel-level result`，

   并且强化 labeling result 上的 probable foreground 和 probable background 作为 未知区域 得到 re-labeling result，在 re-labeling result 上截取一个（仅包含分割前景区域和不一致区域的）矩形区域

4. 在 re-labeling result 的矩形区域上进行 dilation 和 erosion 操作，得到 new-labeling result

5. 使用 new-labeling result 进行像素水平上的 GrabCut 来细化 `superpixel-level result` ，得到 `pixel-level result`



### background connectivity prior (BCP) computation 

BCP 用于量化一个区域连接到图像边界的程度。本文使用 SO [^15] 来计算 BCP。

**SO method:**

1. 使用 SLIC 提取 $N\_s=400$ 个超像素，作为 patches，以 CIELAB 颜色空间上，邻居超像素 $(p,q)$ 的平均颜色欧几里得距离为权重，连接邻居超像素，构造无向带权图。

2. 计算任意两个超像素 $(p,p\_i)$ 之间的几何距离 $d\_{geo}(p,p\_i)$

3. 定义超像素 $(p,p\_i)$ 之间的连接性范围 $S(p,p\_i)$ 为一个高斯分布
   $$
   S(p,p_i) = exp(-\frac{d_{geo}^2(p,p_i)}{\sigma^2_{clr}})
   $$
   其中 $\sigma\_{clr}$ 为高斯分布参数，通过实验取值为 10，$S(p,p\_i) \in [0,1]$ 可以视为像素 $p\_i$ 到 $p$ 的贡献范围

4. 记 $Area(p) =\sum\_i S(p,p\_i)$ 为超像素 $p$ 的区域空间

   当超像素 $p$ 和 $p\_i$ 在一个颜色空间上的 `flat region` 时，$d\_{geo}(p,p\_i) = 0, S(p,p\_i) = 1$ ,表示 $p\_i$ 为 $Area(p)$ 的一个贡献单位区域。

   当超像素 $p$ 和 $p\_i$ 在`different region` 时，将有 $S(p,p\_i) \to 0$ ，表示 $p\_i$ 对 $Area(p)$ 几乎无贡献。

5. 记 $Len\_{bnd}(p) = \sum\_i S(p,p'\_i)$ 为超像素 $p$ 的区域的周长，其中 $p'\_i$ 为任意边界超像素。

   若超像素 $p$ 属于`background region`，则其 $Len\_{bnd}(p) $ 将远大于`object region`的超像素的该值。

6. 定义 BCP 为 $BndCon(p)$
   $$
   \begin{aligned}
   BndCon(p) &= \frac{Len_{bnd}(p)}{\sqrt{Area(P)}}\\
   &=\frac{\sum_{i=1}^{N_s}S(p,p_i)\cdot \delta(p_i \in Bnd)}{\sqrt{\sum_{i=1}^{N_s}S(p,p_i)}}\\
   \delta(flag) & = \begin{cases}
   1, & if \quad flag = Ture\\
   0, & if \quad flag = False\\
   \end{cases}
   \end{aligned}
   $$
   超像素 $p$ 属于 `background region` ，$p^\*$ 属于 `object region`，则有 $BndCon(p)$ 大于 $BndCon(p^\*)$  

7. 将 $BndCon$ 小于 2 的值修改为 0， 再将 $BndCon(p) =0$ 的超像素作为伪前景区

8. 将 $BndCon$ 的值规范法到 $[0,1]$，以表示每个超像素属于背景区域的可能性。



### superpixel-level labeling 

1. 根据伪前景图，使用 Otsu [^27] 算法计算自适应阈值 $t\_m$ 
2. 移除伪前景图中平均显著性值小于 $t\_m$ 的超像素（即认为这些超像素属于（伪）背景区域），得到粗糙前景图
3. 进一步优化 Otsu [^5]，计算阈值 $t\_h$ 用于将`粗糙前超像素`分为 `确定前景超像素` 和 `可能前景超像素`
4. 根据经验，设置阈值 $t\_b$，将BCP大于 $t\_b$ 的超像素标记为`确定背景超像素`，标记其他`粗糙背景超像素`为` 可能背景超像素` 

得到具有四类 labeling seeds



### two-phase GrabCut

**定义无向带权图：**

1. 超像素 GrabCut ：

将超像素作为节点，连接任意两个邻居节点，构成无向图

2. 局部像素 GrabCut ：

通过连接任意邻居像素，构成无向图

在计算过程中限定了一个局部的矩形范围，以提高计算效率

**定义边界平滑项：**
$$
V(x,z) = \gamma \sum_{(i,j)\in Cut}[x_i\neq x_j]exp(-\beta||z_i-z_j||^2)\\
\beta =(2<||z_i-z_j||^2>)^{-1}
$$
其中 $\gamma$ 为相关系数，取常数 0.5 。$Cut$ 是邻居节点集合。$x$ 是表示分割结果的向量，$z$ 是图像的每个像素组成的数据集。$z\_i$ 是节点 $i$ 的颜色值（超像素中取均值），$x\_i$ 是节点 $i$ 对应的分割 label。$\beta$ 为衰变因子及其 $<\cdot>$ 代表一个彩色图像的期望（用作权重值，由图像对比度决定）。

> 这里面的参数β由图像的对比度决定，可以想象，如果图像的对比度较低，也就是说本身有差别的像素m和n，它们的差||zi-zj||还是比较低，那么我们需要乘以一个比较大的β来放大这种差别，而对于对比度高的图像，那么也许本身属于同一目标的像素m和n的差||zi-zj||还是比较高，那么我们就需要乘以一个比较小的β来缩小这种差别，使得V项能在对比度高或者低的情况下都可以正常工作



**高斯混合模型 GMM：**

本文使用颜色量化技术 [^32] 构建 GMM。

GMM 用所有颜色样本初始化单个组件，并且迭代的使用协方差矩阵的特征值和特征向量去计算 划分组件（the component to split）及其分割点（split point）

GMM定义为：
$$
\theta =\{\pi (x,k), \mu(x,k),\sum (x,k), \quad x\in\{0, 1\}, k\in \{1,\cdots,K\}\}
$$
其中 $\pi(\cdot)$、$\mu(\cdot)$ 和$\sum(\cdot)$ 分别为 GMM 的混合加权系数、均值、协方差矩阵。$K=5$ 表示有 5 个组件存在于前景、背景 GMM 模型中。每个超像素\像素 $p\_i$ 都有其对应的分割标签 $x\_i$ 和 GMM 组件索引 $k\_i$

在超像素水平的 GrabCut 中，将`确定前景种子`和`可能前景种子`反馈到前景GMM中，将`确定背景种子`和`可能背景种子`反馈到背景GMM中；在像素水平的 GrabCut 中，则将`确定前景种子`和`非确定种子`（即`可能前景种子`和`可能背景种子`）反馈到前景GMM中，`确定背景种子`和`非确定种子`反馈到背景GMM中。

一旦 GMM 模型建立，就可以使用通过每个超像素\像素 $p\_i$ 得到 $\theta$，然后得到区域项 $U$
$$
U(x,\theta,z) = \sum_{i=1}^N[-\log\pi(x_i,k_i) +1/2\cdot\log\det(\sum(x_i,k_i))\\
+1/2\cdot (z_i-\mu(x_i,k_i))^T\sum(x_i,k_i)^{-1}(z_i-\mu(x_i,k_i))]
$$
$N$ 为颜色样本数量，$\det(\cdot)$ 为行列式。区域项 $U$ 表示每个节点属于前景或背景区域的的可能性

> 区域项 U，表示一个像素被归类为前景或者背景的惩罚，也就是某个像素属于目标或者背景的概率的负对数



**GrabCut求解：**

定义能量函数 $E$ 。

$$
E(x,\theta,z) = U(x,\theta,z)+V(x,z)
$$

使用 EM (Expectation Maximization) 算法训练求解 GMM 模型后，再使用求解最小割（min-cut）算法求得能使能量函数最小化的分割集合 $\hat x$，完成图像的分割

$$
\hat{x}=arg\min_x \min_\theta E(x,\theta, z)
$$
$\hat{x}\_i = 1$ 表示超像素\像素 $p\_i$ 属于前景区域，否则为0，表示属于背景区域



**超像素水平 GrabCut 的不一致性：**

1. 某些由`可能前景种子`标记的区域被分割为背景区域
2. 某些由`可能背景种子`标记的区域被分割为前景区域

为此需要进行像素水平的 GrabCut 来进行细化结果。

**像素水平 GrabCut：**

1. 计算一个矩形区域，其包含了所有潜在对象区域 （即超像素水平分割结果中的前景区域以及于种子标记不符的区域）
2. 将该区域的每个边界向外衍生20个像素，以包含更多的背景信息 
3. 将潜在对象区域的边界向外进行扩张，将超像素分割结果中的前景区域进行侵蚀操作（操作程度多少？），以此来扩大不确定区域（即 origin 表示的区域）的范围，且缩小确定前景区域范围。
4. 潜在对象区域以外的区域用背景种子标记，前景区域侵蚀以内的区域前景种子标记，其他区域使用未知种子标记
5. 使用像素水平 GrabCut 对该矩形区域进行分割，得到最终分割结果





## Experiment 

数据集：MSRA1K , MSRA10K , DUT-OMRON , PASCAL-S , MSRA-B , ECSSD , HKU-IS 和 SOD

比较对象： FT , CB , MA 和 SalCut 

评估标准：F-measure，MAE（平均误差），IoU（intersection over union）



本文使用 4 中显著性模型（RC , MDC , MST , DSS ,）生成显著性图用于 MA 和 SalCut  和本文方法的输入。

### labeling comparison

通过于 MA 和SalCut 的对比，本文算法的 [labeling method](#superpixel-level-labeling) 具有一下优点 :

1. 无论输入的显著性图精度如何，只要其处于 BCP 产生的伪前景区域，均可以用于分割过程
2. labeling method [^3] 对潜在前景区域的标记具有高的recall。然而当显著性图中存在相互分离的显著性对象时，该方法不能标记出所有显著性对象（即只能标记出单个显著性对象）。而本文方法不会出现漏标记情况。

 ### segmentation comparison

SalCut 由于其 labeling 方法只能标记一个，其分割结果只能是一个连通的区域。而其他方法能够分割出所有区域，可与 ground-truth 的相似度不如本文方法高。

通过对比 F-Measure 、MAE、IoU，本文的方法在数值上表现总的来说优于其他算法。
$$
\begin{align}
& F_b = \frac{(1+\beta^2)\cdot Precision\cdot Recall}{\beta^2\cdot Precision+Recall}\\
& MAE = \frac{1}{W\cdot H}\sum_{x=1}^W\sum_{y=1}^H ||S(x,y)-G(x,y)||\\
& IoU = \frac{||S\cap G||}{||S\cup G||}
\end{align}
$$
其中 $\beta^2$ 设置为 0.3 , $W$ ，MAE 中 $H$ 为是 ground-truth $G$ 和 分割结果 $S$ 的宽高 ， IoU 中 $G$ 和 $S$ 分别是ground-truth 和 分割结果的掩图

### ablation analysis

通过进行不同迭代次数（1-3）的超像素 GrabCut 和 不同迭代次数（0-2）像素 GrabCut 的组合实验，从以上三种评价指标的数值上看：

1. 仅含有超像素 GrabCut 的分割结果也是比较良好的，且随着迭代次数的增加，可以略微的提高分割效果，且几乎不会增加时间成本，即验证了超像素 GrabCut 的有效性。
2. 当进行像素水平的 GrabCut 时，分割结果可以有明显的改善，而其迭代次数的增加，仅导致时间成本提高，却不会改善分割效果。

从消融实验中看，本文的两阶段 GrabCut 方法，不仅达到一个较好的分割效果，且在计算效率上也比较高。

### execution time

通过对比 SalCut、MA 和本文方法的执行时间，验证本文方法的计算效率。这三种方法均含有 GrabCut 计算过程。

通过使用不同分辨率 （400 x 300、800 x 600 和 1600 x 1200）图像进行实验，得到一下结论：

本文的超像素水平 GrabCut 时间大致小于 MA 和 SalCut 的 2/3 ，文本的两阶段 GrabCut 时间几乎是 SalCut 的一半，且比 MA 快。验证了本文方法的计算高效性。

### failure case

显著性值较大的非前景区域（即背景与前景的比较相似的区域）被标记为确定前景区种子，且该区域足够大，以至于在 Erosion 操作阶段不能将其划入非确定区域，将导致该区域最终被分割为前景区。



## Annotation

1. superpixel - [请问超像素(Superpixel)的大致原理以及State-of-the-art？ - 知乎](https://www.zhihu.com/question/27623988) 可用于图像降维和去噪
2. GrabCut - [图像分割之（三）从Graph Cut到Grab Cut - zouxy09的专栏 - CSDN博客](https://blog.csdn.net/zouxy09/article/details/8534954)

[^3]: [Global Contrast Based Salient Region Detection](https://ieeexplore.ieee.org/abstract/document/6871397/)
[^5]: [Saliency cuts based on adaptive triple thresholding](https://ieeexplore.ieee.org/abstract/document/7351680/)
[^15]: [Saliency optimization from robust background detection](https://www.cv-foundation.org/openaccess/content_cvpr_2014/html/Zhu_Saliency_Optimization_from_2014_CVPR_paper.html)
[^27]: [A Threshold Selection Method from Gray-Level Histograms](https://ieeexplore.ieee.org/document/4310076)
[^28]: [SLIC superpixels compared to state-of-the-art superpixel methods](https://ieeexplore.ieee.org/iel5/34/4359286/06205760.pdf) 
[^32]: [Color quantization of images](https://ieeexplore.ieee.org/abstract/document/107417/)
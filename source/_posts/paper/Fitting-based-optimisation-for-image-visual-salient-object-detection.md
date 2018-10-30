---
title: Fitting-based optimisation for image visual salient object detection
id: FO_for_SDA
date: 2018-10-26 00:38:03
tags: ['fitting-based', 'saliency detection']
mathjax: true
---


一种对显著性检测算法的优化算法，该优化算法基于拟合。

1. 计算 ground-truth 和显著性图之间的统计信息
2. 使用统计信息计算四个拟合模型的参数
3. 使用拟合模型计算新的显著性图的拟合显著性值，并且限制到合理范围 [0, 255] 。

<!--more-->



> 论文：[Fitting-based optimisation for image visual salient object detection](http://digital-library.theiet.org/content/journals/10.1049/iet-cvi.2016.0027)



---


## Intro 

常用的显著性算法评价指标 precision-recall (RC) curve, receiver operating characteristic (ROC) curving 和 area under the curving (AUC)，它们的一些**主要问题**：

1. 无法区分某些显著性检测算法的性能，因此需要更严格的度量
2. 传统的显著性检测图评价因子有时候不能正确评估显著性图[^1]
3. 传统因子过于看重 true negative saliency detection，即正传统因子将（准确检测显著性对象但是对背景的检测不准确的方法）视为比（准确检测背景但是对某些显著性像素的检测不正确的方法）效果好

本文使用 FR-IQA 解决上述 **显著性评价因子** 的问题。相对于传统评估因子，FR-IQA 考虑更多的因素且更严格。

一些 SDA 基于颜色、纹理、空间距离、形状和其他图像特征，其他 SDA 考虑了图像的背景信息



## The Method

使用 SVO 算法举例。本文方法主要有三个阶段

1. 计算 SVO 显著性图的两个直方图，作为统计信息。一个用于要检测的区域为显著性区域，另一个用于要检测的区域为非显著性区域。然后基于直方图获得独立变量和因变量的点集
2. 基于统计数据，选择合适的方程作为拟合模型，计算模型的参数
3. 对新的 SVO 显著性图使用模型获得拟合显著性值，并且调整到合理范围 [0, 255]，使用拟合值组成结果图

{% asset_img FO_flow.png [FO_flow] %}

### Histogram statistics 

自变量点集：显著性算法得到的显著性图的统计数据

因变量点集：ground-truth 显著性图的统计数据
$$
\begin{aligned}
& H_{mi}^s(k) = \sum_{p \in I}\delta\{ S(p)=k\} \cdot \delta \{G(p)=255\}\\
& H_{mi}^n(k) = \sum_{p \in I}\delta\{ S(p)=k\} \cdot \delta \{G(p)=0\}\\
\end{aligned}
$$

$S(p)$ 是像素 $p$ 的显著性值，$G(p)$ 是像素 $p$ 在 ground-truth 中的像素值，$\delta\\{.\\}$ 当条件满足时取值为 1 ，否则取值为 0 。$H\_{mi}^s(k)$ 即表示（与 ground-truth 中显著区域对应的）显著性算法所得的每张显著性图中的区域上显著性值为 $k$ 的像素点数量。（$G(p) = 255$ 为白色区域，为显著区域）

$$
\begin{aligned}
H^s_m(k) = \sum_{i=1}^{|T|} H^s_{mi}(k)\\
H^n_m(k) = \sum_{i=1}^{|T|} H^n_{mi}(k)\\
\end{aligned}
$$
$H\_m^s(k)$ 为显著区域中显著性值为 $k$ 的像素总个数，$|T|$ 为图像的个数。

拟合点表示为 $(x\_i,y\_i)$ ，其中 $x\_i \in \\{0,1,..255\\}$ 是显著性图中某像素点的显著性值（自变量）；$y\_i\in\\{0, 255\\}$ 是与  $x\_i$ 对应的 ground-truth 中像素的显著性值。点 $(k,255)$ 的数量为 $H\_m^s(k)$，点 $(k,0)$ 的数量为 $H\_m^n(k)$ 。使用缩小因子 $\theta$，得到 $H\_m^s(k) / \theta$ 和 $H\_m^s(k) / \theta$ 。对不同数据集 $\theta$ 按照拟合图片的像素数量取值不同， MSRA-1000 数据集上取值 200，THUS-10000 数据集上取值 1000 。

### Fitting model 

考虑了八个模型类型：distribution, exponential, Fourier function, Gaussian, power function, rational, sum of sines, and polynomial (including linear)，和quality assessment model [^23]。使用数据集的子集来验证每个模型的实用性。最终确定了四个模型为最优的拟合函数：the quality assessment, rational, sum of sines, and linear models。

四个模型均需要满足两个要求：

1. 为连续函数，输出为平滑的saliency map
2. 根据自变量的大小，输出值接近 0 或 255

* quality assessment function 具有五个待定参数 $\beta\_i$
  $$
  f_1(x) = \beta_1(\frac{1}{2}-\frac{1}{e^{\beta_2(x-\beta_3)}})+\beta_4 x + \beta_5
  $$

* rational function 具有四个待定参数 
  $$
  f_2(x) = \frac{\beta_1 x^2+\beta_2 x + \beta_3}{x+\beta_4}
  $$

* sum of sines function 具有六个待定参数
  $$
  f_3(x)=\beta_1 \sin(\beta_2x+\beta_3)+\beta_4\sin(\beta_5x+\beta_6)
  $$

* linear function 具有两个待定参数
  $$
  f_4(x)=\beta_1 x+ \beta_2
  $$





将具有 n 个待定参数的模型记为 $f(x\_i, C\_1,..C\_n)$，使用最小二乘法计算模型的参数
$$
\min \sum_i (y_i - f(x_i,C_1, ..., C_n))
$$
四个模型具有各自的优点和缺点：

1. quality assessment model 单调递增，可以让显著性像素更白，非显著性像素更黑；不足是计算五个参数的时间成本大
2. rational model 和 sum of sines model 为非线性模型，拟合结果比 linear model 好，但是非单调且计算参数时间成本较大
3. linear model 单调且只有两个参数，计算速度快，但拟合结果略差



### Legitimate range 

使用拟合模型对新输入的显著性图计算拟合显著性值，再需要将显著性值调整到合理范围 [0, 255]。调整显著性值的方法需要满足三个条件：

1. 需要维护由拟合模型计算的结果的整体一致性
2. 所有拟合值需要属于 [0, 255] 范围之内
3. 最大最小值分别为 0 和 255

调整方法为具有两个操作

1. 截断操作
   $$
   x_{ti} = \max(\min(\varpi,x_i),k)
   $$
   $x\_i$ 是像素 $i$ 的显著性值，$\varpi$ 取值 254， $k$ 取值 0 。

2. min-max 规范法
   $$
   x_{ni} = \frac{x_i -\min(X)}{\max(X)-\min(X)} \times \varpi
   $$
   $X$ 是显著性图的所有显著性值集合

对于一组显著性值 $X$，若其与 [0, 255] 交集为空或为 $X$ 本身，则只需要执行 min-max 规范法操作；否则即 $X$ 一部分位于 [0,255] ，一部分超出这个范围，此时需要先执行阶段操作，再进行 min-max 规范法操作，以获得最终优化的显著性值。



## Experiments

**数据集**：MSRA-1000 和 THUS-10000

**使用的 SDA**：SMD , SO , VA , MR , HS , RC , HDCT , GS , TD , CB , PCA , LR , SF , 和 SVO 

**评估方式**：传统显著性评价因子（WF , OR, PR–AUC, 和 ROC–AUC,）、IQA 和 CBIR

传统显著性评价因子结果：WF 和 OR 有明显改善，PR-AUC 和 ROC-AUC 几乎无变化

### IQA metrics

采用十二种 FR-IQA：MSE, MAD , SNR, WSNR , PSNR-HVS , PSNR-HVS-M , PSNR, MSSIM , NQM , VIFP , PSNR-HA , 和 PSNR-HMA。其中 MAD 和 MSE 越低表示图像质量高，其余相反。

使用 ground-truth 显著性图作为参考图像，SDA 生成的显著性图作为目标图像。对与 MAD 和 MSE 计算这两指数对优化算法的变化率
$$
R(s,m,d) = \frac{Q(s,m,d)-P(s,d)}{P(s,d)} \times 100\%
$$
$s$、$m$ 和 $d$ 分别表示 SDA 、优化模型和数据集，$P$ 和 $Q$ 分别代表原始显著性算法和拟合优化的显著性算法。再对于所有 SDA 的平均变化率 
$$
R(m,d) = \frac{R(s,m,d)}{n_s}
$$
其中 $n\_s$ 为 14 ，表示实验中使用了 14 种 SDA。

结果： MSE 和 MAD 都有明显减少，且在两个数据集上，其余指数都有明显提高，且 quality assessment 模型的提高最显著。且从算有因子上看，quality assessment 模型改善效果最明显，linear 模型改善效果最弱。时间成本上，quality assessment 模型耗时最多，linear 模型耗时最少。



### CBIR application

CBIR 根据图像的内容相似度进行搜索，使用 显著性图 作为 权重图。在 CBIR 中本文使用 RBG 通道距离衡量两张图的相似度。

**CIBR 实现步骤**：

1. 使用显著性图获得图像的加权 RBG 直方图
2. 计算两个图像加权 RBG 直方图之间距离作为相似度
3. 根据相似度得到最相似的指定数量的图像

在 RBG 直方图中，使用 8 bins 表示一个像素（像素取值0~255），故直方图还有 $8^3=512$ 个 bins。 第 $i$ 个 bin 的像素颜色取值范围为 $b\_i$，图像 $I$ 的直方图 $l$ 为
$$
l(i)= \frac{\sum_{p\in I} \delta\{I(p)\in b_i\}}{W\times H}
$$
加权直方图 $h$ 为
$$
h(i)= \frac{\sum_{p\in I} M(p)\delta\{I(p)\in b_i\}}{W\times H}
$$
图像对应直方图的相似度计算
$$
fc(I_1,I_2)=\sum_{i=1}^q \min(h_1(i), h_2(i))
$$
其中 $q$ 取值 512 代表 bins 数量（x轴长度），$fc$ 与 两图像的相似度成正相关。



**CBIR 应用于评价**：

对应每个 SDA的评价步骤

1. 获取数据集中每个图的 ground-truth 显著性图、SDA 显著性图、优化后的 SDA 显著性图
2. 计算分别由使用SDA著性图和 ground-truth 显著性图得到的检索结果图之间的 SROCC ，（SROCC 表示图像之间的相似度，用于衡量 SDA 表现 ） 
3. 同样的方式计算获取优化后的 SDA 的 SROCC
4. 对比 SROCC 获得优化模型的优化效果

根据 SROCC 的对比及其变化率，可得优化模型改善了 CBIR 的效果，且 quality assessment 模型效果最好。



### Discussion

对于只检测到显著性对象的一小部分的显著性图，难以将其优化为检测到完整的显著性对象的显著性图。



### Personal understanding

本文的优化算法实质上是对强化原显著性图的效果（非显著区域更加非显著，显著区域更加显著），因此若 SDA 计算出的显著性图存在根本上的错误，则本文优化算法无法起到纠正作用。






## word

FR-IQA：full-reference image quality assessment

RMSE：root mean square error，$\sqrt{\frac{\sum\_{t=1}^n(\hat{y\_t} - y\_t)^2}{n}}$，[均方根误差](https://zh.wikipedia.org/wiki/%E5%9D%87%E6%96%B9%E6%A0%B9%E8%AF%AF%E5%B7%AE)

CBIR：content-based image retrieval

SDA：salient object detection algorithm

SROCC：Spearman's rank order correlation coefficient

[^1]: [How to evaluate foreground maps](https://ieeexplore.ieee.org/abstract/document/6909433)

[^23]: [A Statistical Evaluation of Recent Full Reference Image Quality Assessment Algorithms](https://ieeexplore.ieee.org/abstract/document/1709988)
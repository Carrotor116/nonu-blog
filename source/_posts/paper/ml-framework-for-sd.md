---
title: Machine learning-based framework for saliency detection in distorted images
id: ml_framework_for_sd
date: 2018-10-30 15:37:48
tags: ['saliency detection', 'distorted imgaes', 'unresolved issues']
mathjax: true
---


目前主要的显著性检测算法是针对无失真的图像进行的，本文提出了一种基于机器学习的 **显著性检测框架** 用于对不同失真类型和水平的图像进行显著性检测。

<!--more-->

> 论文：[Machine learning-based framework for saliency detection in distorted images](https://link.springer.com/article/10.1007/s11042-016-4128-1)


---
## Intro 

图像的失真类型有噪声、污点和压缩等，失真会改变图像的低级特征，进而降低显著性检测的效果。

目前多数显著性检测方法假设图像是无失真的。论文 [^17] 提出了一种基于非参数回归框架的方法用于噪声图像的显著性检测。

本文贡献：

1. 评估了目前最先进的一些显著性检测算法在不同失真类型和失真水平下的表现。结果表示不同失真的影响不同，总的来说失真降低的显著性检测的效果，尤其是高的失真水平
2. 以噪声和 JPEG 压缩为例，提出针对失真图像的显著性检测框架
3. 提供了一个新的数据集 [TID2013S](#TID2013S) 用于进行失真图像的显著性检测



## TID2013S

TID2013 是广泛用于 IQA 的数据集，包含 25 张参考图，3000 张失真图像，失真图像对应每个参考图的 24 种失真类型以及每类型失真的 5 个失真水平 （$25 \times 24 \times 5 = 3000$）。失真图像的 PSNR 比参考图像低，使用 PSNR 来模拟失真水平的表达，五个失真水平对应的 PSNR 为 33、30、27、24 。



TID2013S 基于 TID2013 数据集得到，在 TID2013 基础上，包含了 25 张参考图的 ground-truth 显著性图。

ground-truth 显著性图的获取方式：

1. 让多位（5位）用户使用矩形标注出图像中的显著性对象，并且选择多数用户标注的结果。

2. 沿着对象的轮廓人工标注显著性对象，使用 `Adobe Photoshop CS6` 进行人工标注，白色区域代表显著性对象，黑色区域代表背景。



### Performances analysis of SDA

评估分析 15 种最最先进的 SDA 在失真数据集上的表现，使用 PR-AUC 来衡量算法，其中精确度定义为正确检测的像素与所有检测像素的比值，recall 值定义为正确检测的像素与 ground-truth 显著性像素像素的比值
$$
\begin{gather*}
Precision = \frac{TP}{TP+FP}\\
Recall = \frac{TP}{TP+FN}
\end{gather*}
$$
其中 $TP$ 为 true positives （检测为显著性像素且正确）的数量，$FP$ 为 false positives （检测为显著性像素而实际并非显著性）的数量，$FN$ 为 false negatives （检测为非显著性像素而实际为显著性像素）的数量。

实验结果显示，14 种 SDA 中除了 DA 算法以外，其失真图像的平均 PR-AUC 均小于参考图像的，且随着失真水平的提高，PR-AUC 减小的更多。



## The Method

主要框架：

1. 使用 RFC 训练失真水平预测模型
2. 使用模型预测 测试图像 的失真水平
3. 对应失真类型和失真水平设置 DRA 的参数，并且对失真图像进行去失真处理
4. 使用 SDA 计算去失真图像的显著性图

失真水平预测模型的训练：

* 特征：计算整个图像和局部图像区域的失真大小（distortion sizes）

* 标签：该图像的失真水平（distortion level）

* 模型： RFC 模型。



### Framework in noisy images

提出了一个基于机器学习的噪声水平预测方法：

1. 使用论文 [^24] 计算图像中每个像素的的噪声值
2. 将图像划分为 $3\times 3$ 的网格，计算整个图像和 9 个单元的评价噪声值记为 $N\_0,N\_1,..,N\_9$，作为噪声范围
3. 将噪声范围排序，将排序后序列作为机器学习的特征，训练预测噪声水平

根据所预测的噪声水平设置最佳的降噪参数，使用高斯滤波器（Gaussian filter）进行降噪，然后计算显著性图。



### Framework in JPEG compressed images

1. 将图像划分为 $3\times 3$ 的网格
2. 使用论文 [^30] 中的方法计算整个图像和 9 个单月的 JPEG 压缩值，$C\_0,C\_1,..,C\_9$，作为压缩范围

3. 将压缩范围排序，将排序后序列作为机器学习的特征，训练预测 JPEG 压缩水平

根据所预测的压缩水平设置最佳的参数，使用解压缩算法 TV [^5] 对图像进行解压，然后计算显著性图。





## Experiment 

### Performance in noisy images

数据集：TID2013S 中 24 种失真类型中的 10 中噪声失真类型；MSRA 1000 dataset

SDA：5 种表现相对较好的显著性检测算法，DR, VA, MR, SO, 和 GS

#### without de-noisy

未使用降噪算法的实验：噪声水平 1-3 对 SDA 的影响相对较少，噪声水平 4,5 对 SDA 的负影响比较大

#### with Gaussian filter and default setting

使用 Gaussian filter 和默认设置参数（template size of 3 x 3，sigma of 0.5）的实验：

除了类型 #9 和 #6 外各噪声等级下，平均 PR-AUC 有所改善，尤其是在 level 5 的噪声水平上；而是在类型 #6 和 #9 数据上的改善很小。

#### with Gaussian filter and optimal setting

使用 Gaussian filter 和最佳设置参数的实验：

template size 为 $\\{3\times 3,5\times 5,7\times t\\}$，sigma 为 $\\{0.5,0.7,0.9\\}$。

针对 VA 和 MR 算法进行实验，选择最优的 template size 和 sigma 组合（VA-best 和 MR-best）来设置 Gaussian filter，以使得这两个算法各自在各个噪声等级下得到最大的平均 PR-AUC 值。从实验结果上可看出，VA-best 和 MR-best 下的平均 PR-AUC 均有进一步改善。

模型训练阶段，所有噪声等级的图像随机分入5组，按 4：1 作为训练集和测试集。训练噪声水平预测模型，得到的模型其准确率为 61.86%，其 close prediction rate 达到 90.56%。

本文框架中结合 VA、MR 算法，设置最优参数 VA best-M 和 MR best-M 后，可使得 VA 和 MR 在各噪声等级下均达到最高的平均 PR-AUC 值。

<span style="color: #f00">**problem：**<span>

* VA best-M 和 MR best-M 具体参数如何获取，竟然没提及？？？！！
* 噪声预测模型如何用于选择最优参数 VA best-M 和 MR best-M 竟然没有提及？？!



### Performance in JPEG compressed images

设置压缩值 $k\in\\{1,2,3,4,5,6\\}$  和 max primal-dual gap $\tau \in \\{0.1,0.5,1\\}$ ，从二者的 18 种组合中选择最优组合（即能使在所有压缩等级下平均 PR-AUC 达到最大的组合），得到 $k=5,\tau=1$ 

进一步进行实验，针对 VA 和 MR 算法，寻找在各个压缩等级下，对应的最优参数组合 $k$ 和 $\tau$，如下

|         | level 1 | level 2 | level 3 | level 4 | level 5 |
| ------- | ------- | ------- | ------- | ------- | ------- |
| VA best | 1,0.1   | 1,0.1   | 4,1     | 2,1     | 3,0.5   |
| MR best | 5,0.5   | 1,0.1   | 5,0.5   | 1,0.5   | 1,0.1   |
| Best    | VA best | MR best | VA best | VA best | VA best |

其中 Best 的参数组合可以明显改善显著性检测算法的 PR-AUC 值。



类似噪声水平预测模型，训练压缩值预测模型，精度达到 89.28% 和 close prediction rates 达到 99.0% 。

在 VA best-M 和 MR best-M 参数组合下，VA 和 MR 可在各个压缩值（表示失真水平）的数据下得到最大平均 PR-AUC 。

<span style="color: #f00">**problem：**<span>

* 同样未说明如何获取 VA best-M 和 MR best-M，以及这两组参数与压缩值预测模型之间的关系。



## Personal Understanding

本文提出用于改善显著性检测算法在失真图像下的表现。具体方法是先通过将失真图像进行去失真（如降噪或解压缩），再将去失真图像用于显著性检测。在去失真过程中，使用的是现有方法( Gaussian filter 和 TV [^5] ），显著性检测算法也是使用的现有算法（VA 和 MR 等）。

本文框架重点在于通过预测失真水平，设置去失真过程的参数，以最大化去失真效果，改善显著性检测的效果。<span style="color: #f00">**然而**，本文却未明确说明 **如何使用预测的失真水平来设置去失真参数**，即 VA best-M 和 MR best-M 。</span>





## Annotation 

TID2013 : [TAMPERE IMAGE DATABASE 2013 TID2013](http://www.ponomarenko.info/tid2013.htm)

IQA : image quality assessment

SDA : saliency detection algorithms

DRA : distortion removal algorithms

PR-AUC : [Differences between Receiver Operating Characteristic AUC (ROC AUC) and Precision Recall AUC (PR AUC)](http://www.chioka.in/differences-between-roc-auc-and-pr-auc/) 和 [PR曲线，ROC曲线，AUC指标等，Accuracy vs Precision - blcblc - 博客园](https://www.cnblogs.com/charlesblc/p/6252759.html)

[^5]: [A Total Variation–Based JPEG Decompression Model](https://epubs.siam.org/doi/abs/10.1137/110833531)
[^17]: [Visual saliency in noisy images](https://jov.arvojournals.org/article.aspx?articleid=2121642)
[^24]: [What Makes a Professional Video? A Computational Aesthetics Approach](https://ieeexplore.ieee.org/abstract/document/6162974/)
[^30]: [No-reference perceptual quality assessment of JPEG compressed images](https://ieeexplore.ieee.org/abstract/document/1038064)




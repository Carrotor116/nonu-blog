---
title: Disentangling Structure and Aesthetics for Style-Aware Image Completion
id: Style-Aware-Image-completion
date: 2018-09-20 08:46:56
tags: ['style-aware', 'image completion', 'image']
mathjax: true
---

一种基于AIC的图像修复方法，考虑图片的结构信息和视觉风格信息来选取patches 并且对其进行自适应风格化，从而提高图像修复效果。

<!--more-->

- 定义图片的结构和艺术风格的计算标准
- 通过视觉搜索（visual search）从数据集和获取候选补丁集合
- 定义能力函数，使用MRF算法在候选补丁集合上迭代，获取所选补丁集合
- 对所选补丁集合进行NST转化，得到自适应风格的补丁集合
- 使用梯度域混合算法将自适应风格的补丁集合合并到待修复图片上

> 论文：[Disentangling Structure and Aesthetics for Style-Aware Image Completion](http://openaccess.thecvf.com/content_cvpr_2018/html/Gilbert_Disentangling_Structure_and_CVPR_2018_paper.html)

------



## Style-aware Image Completion

### Disentangling Patch Structure and Aesthetics

使用两个三元组卷积神经网络进行特征嵌入（[style embedding](#style-embedding) & [structure embedding](#structure-embedding)）。
$$
\{g_{s}(p), g_{z}(p)\}
$$
$p$ 表示一个图片补丁。

$g\_{s}$ 学习图片内容语义不变的情况下的美术风格相似度

$g\_{z}$ 学习风格不变下的结构相似度

这两个神经网络使用同构设计，包含所有分支的GoogLeNet骨干，且pool5层之后附加了一个共享所有权重的低维度(128-D)瓶颈层。特征嵌入在瓶颈层获取。

#### style embedding

由于需要在多种分辨率上挑选并风格化补丁，所以需要能使用多规模下的特征嵌入。学习一组style embeddings
$$
S = \{g_{s}^{l}\}, l = [0, L]
$$
这组 style embedding 表示方形补丁小的半个八度音程（half-octave）间隔，从40像素到60像素。	

我们通过判别（softmax）损失从头开始训练每种风格，作为给定一组手工标注样式类别的艺术图片和照片的分类器。

**数据集**：从`BAM!`中随机选取的88千张图片，包含了8中风格：水彩、矢量艺术（vectorart）、3D、石墨、铅笔画、优化、漫画、照片。

对风格网络使用三重损失硬负采矿（hard negative mining）进行微调，三次损失分别为 anchar $\alpha$ 、含有相同艺术风格不同物体内容的**正分支** $p$、含有不同艺术风格和相似物体的**负分支** $n$。

损失函数 $\zeta$ 为：
$$
\zeta(\alpha, p, n) = [m+|g_{s}^{l}(\alpha) - g_{s}^{l}(p)|]^{2}- |g_{s}^{l}(\alpha) - g_{s}^{l}(n)|^{2}]_{+}
$$
$m=0.2$ 为与收敛相关的参数，$[x]\_{+}$ 是向上取整，

#### structure embedding

艺术风格不变下的结构嵌入 $g\_{z}(.)$ 即是structure embedding。计算与style embedding相似，不同之处为**正分支**和**负分支**与其相反。

且数据集不同，由于目前的`BAM!`只包含9个语义类别注释，为了促进对艺术作品的语义概括，网络在更广泛的数据集上受到进一步的微调阶段，该数据集来自Behance网站（从中衍生出BAM）



### Patch Aggregation and Selection

#### Style and Structure aware Retrieval

数据集：Behanec（一个创意专业人士的网站）上的 66.8M 张用户产生的图片

每张图片根据 $g\_{z}$ 和 $g\_{s}^{0}$ 计算一个描述符
$$
I(d) = PQ([g_{z}(d) g_{s}^{0}(d)], B)
$$
$PQ(.)$ 代表产品量化[11]。使用PQ扩张数据集的图片，基于`k-NN`在 $||I(s)-I(d)||\_{2}$ 上搜索返回前200张图片得到检索结果，即**候选补丁集合**，记为P。



### Patch Selection over Learned Embeddings

将图片上缺块划分为网格，每个格子与其邻居的一般重叠。将填充缺块视为标签问题。使用MRF优化算法从候选补丁集合P中选择最优子集合。该优化算法能平衡补丁的选择来最小化内容结构、风格、外观（空间连贯性），从而最小化能量函数 $E$ 。
$$
E(X)=\sum_{i\in\mu}\psi_{z}(p_{i}) +\frac{1}{N_{i}}\sum_{i\in\nu,j\in N_{i}}\psi_{ij}(p_{i}, p_{j})+\sum_{i\in\nu}\psi_{s}(p_{i})
$$
$\nu=\\{\nu\_{1}, ..,\nu\_{n}\\}$ 为对应的所有网格的集合。 $p\_{i}\in P$ 表示第 $i^{th}$ 个网格 $\nu\_{i}$ 的补丁标签。$N\_{i}$ 为网格 $\nu\_{i}$ 四周的邻居网格集合。
$$
\psi_{z}(p_{i}) = ||g_{z}(p_{i}) - g_{z}(s)||_{2}
$$
一元函数 $\psi\_{z}(p\_{i})$ 表示补丁 $p\_{i}$ 与原图之间的结构偏差，使用 $L\_{2}$ 距离表示。

二元函数 $\psi\_{ij}(p\_{i}, p\_{j})$ 测量与邻居网格之间的空间连贯性，通过补丁重叠部分的像素值的平方差之和表示。[23] 
$$
\psi_{s}(p_{i}) =|g_{s}^{l}(p_{i}) - g_{s}^{l}(s) |+\frac{1}{N_{i}}\sum_{p_{j}\in N_{i}} |g_{s}^{l}(p_{i}) - g_{s}^{l}(p_{j})|
$$
第三项 $\psi\_{s}(p\_{i})$ 用于促进图片局部的风格连贯性，使用含style embedding 的 $L\_{2}$ 距离表示。

所以最小化能力函数 $E(X)$ 表示促进各个边缘信息的空间连贯性（第二项）和局部风格连贯性（第三项），确保局部语义分布的相似性（第一项）。

能力函数与[15]相似，也具有采取补丁的加权平均的一元潜力形式
$$
E(X)=\sum_{i,j\in\nu}(\psi_{z}(p_{i})+ \psi_{z}(p_{i},p_{j}))
+ \sum_{i\in\nu,j\in N_{i}}(\psi_{ij}(p_{i},p_{j}))
$$
MRF 在多规模 $l=[0, L]$ 上**迭代求解**的，规模与style embeddings $g\_{s}^{l}(.)$集合对应。

### Adaptive Patch Stylization

本文中的算法在将所选补丁集合 $X$ 合并入原图 $s$ 前，对补丁集合进行自适应的风格化。采用神经风格转化`NST`[7]。抽取补丁的结构描述符 $\chi\_{z}(p\_{i})$ 和风格描述符 $\chi\_{s}(p\_{i})$ 。计算修正后的补丁 $p'\_{i}$ ，其中 $\chi\_{s}(p') \simeq \chi\_{s}(s)$，且 $\chi\_{z}(p') \simeq \chi\_{z}(p\_{i})$ ，即风格和原图相似，结构与原补丁相似。其风格的一致性是由MRF的第三项驱动的：
$$
\zeta_{sty}(p'_{i}) = |\chi_{z}(p'_{i})-\chi_{z}(p_{i})|-\alpha e^{-\psi_{s}(p_{i})}|\chi_{z}(p'_{i})-\chi_{z}(s)|
$$
$\alpha = 10^{-5}$ 是平衡结构和风格的规模标准化术语。$p'\_{i}$ 的初始化是 $p$  加上高斯噪声 、损失项 $\zeta\_{ssy}$ ，并且通过 ADAM 最小化。

根据[7] $\chi\_{z}(.)$  是通过训练前的VGG-19采样conv\_4层的forward-pass 来获取的，$\chi\_{s}(.)$ 通过粗风格嵌入 $g\_{s}^{{0}}(.)$ 

因此`风格化`的 "强度" 被由MRF优化算法决定的风格相似度所控制。风格化后的补丁通过梯度域混合算法[21] 组合到原图 $s$ 

最后风格化的补丁（stylized patches）使用梯度域混合算法[21] 组合到图片中。

## Experiments and Discussion

**数据集**：Places2 [32]，通常用来修复图片的数据集；采样自`BAM!`数据集的修复图片的数据集 [30]

**评价因子**：主观的用户研究、SSIM [27]、SWD [13]

**比较对象**：PatchMatch [1]，Image Melding [3]，Million Image [8]，ContextEncoder [20]

**实验结果**：实验结果表明，在SSIM和SWD指标中本文的方法效果明显优于其他对比算法，且在算法的耗时方面也具有明显优势。并且还进行了消融实验，验证了本算法中图像结构化因素、风格因素以及自适应风格化三者各自的独立有效性。
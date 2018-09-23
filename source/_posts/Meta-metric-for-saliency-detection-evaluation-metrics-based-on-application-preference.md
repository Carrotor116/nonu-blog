---
title: >-
  Meta-metric for saliency detection evaluation metrics based on application
  preference
id: meta-metric-for-sdem
date: 2018-09-23 09:28:36
mathjax: true
tags:
---

本文提出了一种基于应用偏好对 SDEM 进行评价（选择）的框架

<!-- more -->

不同 SDA 在不同类型应用上的影响表现是不同的，因此选择为应用选择合适的 SDA 是重要的。这引出了 SDEM 。由于不同 SDEM 对同一个 SDA 的评估结果具有不一致性，且现有 SDEM 未考虑 SDA 在应用中的表现。为了消除这些 SDEM 评价结果的不一致性影响，本文提出一个基于应用的 meta-metric 来评估 SDEM。	本文的 meta-metric 基于特定应用 **CBIR** 。

> 论文：[Meta-metric for saliency detection evaluation metrics based on application preference](https://doi.org/10.1007/s11042-018-5863-2)



## 概述

* 根据 $SDA\_{k}$，对数据集中每个图片进行 [Saliency weighted CBIR](#Saliency-weighted-CBIR)，得到 $L\_{k}$
* 计算 $mAP\_{k}$ 得到 $SDA\_{k}$ 在CBIR 中的准确度
* 将所有 $mAP\_{k}$ 进行排序得到 SDA 在CBIR 中的表现排序 $\hat{X}$
* 使用 $SDEM\_{b}$ 计算 $SDA\_{k}$ 的 $SM\_{avg}$，得到 SDA 的评估排序 $X\_{b}$
* 计算 $X\_{b}$ 和 $\hat{X}$  的相关系数 $Y\_{b}$ 作为 $SDEM\_{b}$ 的评估值



## Saliency weighted CBIR

* 从 ResNet-152 的 res5c 层中每张图抽取 DCF [14]，其包含 $C=2048$ 个 feature map

* 使用  $SDA\_{k}$ 生成每张图的 SM 

* 计算每张图的 $SWD\_{k}$。按空间位置 $(x,y)$ 关联 SM（$sal$） 和每个 DCF（$f$），采用简单的加权启发式结合二者，$H$、$W$ 为 SM 的高、宽，$\psi\_{1}(I)$ 的维度为 $C=2048$
$$
  \psi_{1}(I) = \sum_{y=1}^{H}\sum_{x=1}^{W} sal (x,y)* f(x,y)
$$

* 使用 $L2$ 距离将 SWF 标准化，得到 $\psi\_{2}$
$$
  \psi_{2}(I) = \frac{\psi_{1}(I)}{||\psi_{1}(I)||_{2}}
$$

* 使用 PCA 压缩和白化处理 SWF 标准化结果，得到白化向量 $\psi\_{3}(I)$
* 将白化向量标准化，得到 SWD $\psi\_{4}(I)$

* 通过计算两张图的 SWD 的标量积，得到两张图的相似度

$$
sim (I_{1}, I_{2}) = <\psi_{4}(I_{1}),\psi_{4}(I_{2})>
$$

* 根据数据集图片与查询图片 $I\_{i}$ 的相似度，得到图片 $I\_{i}$ 的**检索结果有序列表**

$$
q_{i} = \{l_{1}, l_{2}, ...l_{N-1} \}\quad i=1,2,..,N
$$

采用 $SDA\_{k}$，对每张图进行上述 CBIR，得到 N 张图片的检索结果有序列表，记 $L\_{k}$ 为SDA $k$ 的**检索结果有序集合**
$$
L_{k} = \{q_{1},q_{2},..,q_{N}\}
$$
将数据集中提供的每个图片的检索有序列表的集合，记为 $\hat{L}$ ，（即 `ground-truth`）



## 评价 SDEM

### 计算 SDA 平均精度

$$
mAP_{k} = \frac{\sum_{i}AP_{k}(q_{i})}{N},i=1, 2,..,N \\
APk(q_{i}) = \frac{\sum_{l_{j}\in \hat{L}(qi)}Precision (l_{j},L_{k}(q_{i}))}{|\hat{L}(q_{i})|}，j=1,2,..,|\hat{L}(q_{i}|
$$

$l\_{j}$ 的 `ground-truth` 中对应搜索图片 $I\_{i}$ 所得的有序列表中 第 $j$ 个图片。$mAP\_{k}$ 越大，$SDA\_{k}$ 越好

计算每个 $SDA\_{k}$ 的 $mAP\_{k}$ ，得到 $\{mAP\_{k} |k=1,2,..N\}$，本文中取 $N=10$ 种SDA，排序后得到各 SDA 在 CBIR 中的表现顺序，记 $\hat{X}$



### 计算 SDEM 相关系数

使用 $SDEM\_{b}$ 评估 $SDA\_{k}, (k\in\{1,..10\})$ 的每张$SM$，取其平均值，排序 $k$ 个平均值得到 $X\_{b}$，计算 $X\_{b}$ 和 $\hat{X}$ 的关系系数 $Y\_{b}$
$$
Y_{b} = 1-\frac{6*\sum_{k}(\hat{X}(k)-X_{b}(k))^{2}}{(n+1)*n*(n-1)}
$$
其中 $k=1, 2...10$ 表示本文中采用了 10 种 SDA，$b=1,2,..,24$ 表示本文中使用的 24 种SDEM。对 $Y\_{b}$ 进行排序，可得到对于 CBIR 最合适的 SDEM。



## 实验

**数据集：**三个， MSRC [33], COSDATA [16], and IID [31]， Oxford Building dataset [27]作为对比时用
**SDA：**十种， HS [18], MST [38], CA [13], FT [1], RC [6], SMD [24], TD [7], MR [44], SO [48], and BSCA [45]
**SDEM：**四种传统的（PR-AUC, ROC-AUC, OR,and WF ），20 个 FR-IQA（...）
**对比对象：**[22] 中提出的 meta-metric 

**实验结果：**本文因子计算出**ROC-AUC**为CBIR中最佳的SDEM，而 [22] 中**WF**为最佳的因子，故通过对比**ROC-AUC**和**WF**来评价本文的 meta-metric 和[22]中的 meta-metric 。使用 Oxford Building dataset 进行对比。ROC-AUC 与 Oxf_result（ground-truth）之间的 SROCC 为 0.81，大于 WF 和 Oxf_result 之间的 SROCC 0.52，故本文的meta-metric 更可靠。而且 24 种 SDEM 使用该数据集上进行 CBIR 得到的表现排序 Oxford_ranking 和本文的 meta-metric 计算出的在其他三个数据集上的 average_ranking 之间具有 0.8574 的SROCC，进一步表示本文 meta-metric 的可靠性。



## 术语
SDA：saliency detection algorithm 显著性检测算法
SDEM：saliency detection evaluation metric 显著性检测评价因子
mAP： mean average precision 平均精度。 测量检索结果准确度的数值
CBIR：content-based image retrieval 基于内容的图片检索

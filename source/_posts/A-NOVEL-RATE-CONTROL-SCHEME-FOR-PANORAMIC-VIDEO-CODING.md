---
title: A NOVEL RATE CONTROL SCHEME FOR PANORAMIC VIDEO CODING
date: 2018-08-15 23:40:24
tags: 论文翻译
---
## abstract 
<!--more-->

Fig.1: 用于参考和修复序列的球面到平面投影的照明。这样的投影导致了在平面领域计算S-PSNR时采样的不均匀。

## 1. 介绍

## 2. 初步
  由于我们的RC模式主要致力于优化S-PSNR，我们将在2.1节简要回顾S-PSNR，并在2.2节验证它的有效性。

### 2.1 S-PSNR概要
  S-PSNR主要在于测量给出参考视频的受损全景视频的失真(distortion)。与PSNR不同，S-PSNR是根据在球面区域的平均方差(MSE)，计算S-PSNR时需要这样的球面投影。公式(figure)1显示了参考和受损序列的球面投影，公式中用于计算逐象素(pixel-wise)S-PSNR的平面区域像素是不平均采样的。
  更加具体的说，我们假设球面上的像素代表`({g(n)}^N)n=1`，其中`N`是全景视频球面上的像素总数。并且，在球面上的g(n)是参考和受损序列上的投影位置`(xn，yn)`。给定`(xn，yn)`，S-PSNR可以根据平均球面平均方差(`S-MSE`)计算，如下。
{% asset_img formulate_1.png  %}

### 2.2 验证S-PSNR 
  为了验证S-PSNR的有效性，我们对比了PSNR和S-PSNR的表现，通过评估他们在几个全景视频序列中相关的主观质量。为了对比，在IEEE 1857 工作组[17]的标准测试集的原始格式中挑出8个全景视频序列。这些序列在500Kbps、3Mbps和100Mbps上被压缩。然后，对比参考原始序列，在压缩序列上测量PSNR和S-PSNR。  接下来，为了根据DMOS来量化那些压缩序列的主观主流，我们按照[18]的方法进行主观测试。在我们的主观测试中，有17个主体（包括10个男性和7个女性，年纪在19到30岁之间）参与评估了所有序列的的原始质量分数。这里，VR头戴式耳机Oculus Rift DK2被用来以原始分辨率(original resolution)播放测试全景视频。另外，我们利用LiveViewRife作为Oculus Rift视频播放，并采用它默认的球面投影类型。最后，在他们基础的评估主观得分上（rated subjective score），每个压缩序列的DMOS数值可以被计算出，如表1中呈现的。表1也显示了每个序列的PSNR和S-PSNR。  现在，我们对每个压缩序列测量PSNR/S-PSNR和DMOS之间的相关性，使用皮尔逊相关系数(PCC)，斯皮尔曼等级顺序相关系数(SROCC),根MSE(RMSE)和平均绝对误差(MAE)。所有压缩序列的相关结果显示在表2中。观察表2，对比PSNR，S-PSNR与DMOS更相关。此外，DMOS和S-PSNR之间的错误比DMOS和PSNR之间的错误更小。因此，S-PSNR是比PSNR更合理的客观指标，证明了S-PSNR的有效性。  

## 3 提出率控制模式 (proposed rate control scheme) 
  RC在全景视频编码的主要目标是为了在给的的比特率上最大化S-PSNR，由于前述证实了S-PSNR的有效性。根据S-PSNR在公式(1)的定义,基于球面的失真是在球面上进行像素采样平方误差的总和(the sum of square error bettween pixels sampled from sphere)： 



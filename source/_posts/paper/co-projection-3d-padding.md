---
title: co-projection-plane based 3-D padding method
id: co-projection-3d-padding
category: paper
mathjax: true
date: 2018-11-08 09:46:45
tags: ['co-projection', '360 video']
---

本文提出一个基于共投影平面的填充方法，利用邻居平面的像素信息，对平面进行扩张填充以维持全景视频二维映射的纹理一致性。

<!-- more -->

> 论文： [Co-projection-plane based 3-D padding for polyhedron projection for 360-degree video](https://ieeexplore.ieee.org/abstract/document/8019393)

---

## Intro

为了适应已有的编码标准（如 AVC 或 HEVC），全景视频通常会映射为 2D 格式以便于压缩处理。`多面体映射方式` 具有较小的几何失真，可是在平面边界处**存在有明显的纹理不一致性**。该不一致性由多面体展开成平面时产生，或由于将球面映射到多面体的不同平面而产生。当 motion vector (MV)  发生于边界时，motion compensation (MC) 将**导致不合理的预测块（块效应），导致严重的编码质量下降**。

为解决平面边界纹理不连续问题，本文提出基于共投影平面的填充方法，利用邻居平面的像素信息，对平面进行扩张填充以保证纹理的一致性。



## Polyhedron Projection

以六面体为例

{% asset_img polyhedron_projection.png %}

使用六面体包围内切球体，以圆形作任意直线ON，交球面于点 M，交六面体于 N 点。则将球面点 M 的像素映射到 六面体上的 N 点。由于 M 可能不属于整数采样点，需要使用插值法得到点 M 的像素值。可使用 Lanzcos [^Lanzcos ] 插值法，对亮度分量使用 Lanzcos2 插值过滤器，对色彩分量使用 Lanzcos3 插值过滤器。

得到投影的立方体后，将立方体展开，即可得到立体映射的平面结果。通常有 4 x 3 和 3 x 2 两种展开方式。

{% asset_img unfold_cubic_format.png %}





## The Method

本文以六面体映射中的一个面（右面）为例说明。



### Approximated texture continuity

对球面进行六面体中的右面及其邻接面进行映射并展开，二维映射结果 Fig.4 ；再对顶面和底面进行旋转对齐操作，达到大致的纹理一致性 Fig.5。

{% asset_img example1.png %}



### Exact texture continuity

右面的邻接面的边界存在纹理的不一致性。通过利用邻接面像素信息，对右面进行拓展填充。如 Fig.6 中，通过把邻接点中点 H 的像素信息映射到点 T ，实现将平面 `ABCD ` 扩张为 `A'B'C'D'` 。

{% asset_img co-projection-plain.png "Fig.6 co-projection-plain" %}


设点 `A'` 为原点 (0,0)，`T` 为 (x,y)，扩张的程度为 m （即边 `MC` 长度为 m），六面体边长为 a，则有
$$
\begin{aligned}
TK = \frac{a}{2}+ m -y\\
JK= x-a-m
\end{aligned}
$$
进而可得到线段 `HS` 的长度
$$
\begin{aligned}
TK = \frac{ST}{O'T}\times OO' = \frac{JK}{O'K} \times OO'
\end{aligned}
$$
同理可得到线段 `SJ` 长度
$$
SJ = \frac{O'J}{O'K}\times TK
$$
以点 `A'`、`O'` 和 `T` 为条件，即可得到点 `H` 的坐标表达式，故对任意点 `T`，可以得到点 `H` 的坐标，实现使用邻接面像素的映射。由于计算得 `T` 的坐标可能非整数，同样需要使用插值法得到其像素值。另外，线段 `A'A` 、`B'B` 、`C'C` 和 `D'D` 上的点被映射到邻接面的边界上），而邻接面的边界之间同样存在纹理差异，若使用插值法将得到不合理的像素值，故这四条线段中的点，使用其周围点的像素平均值来进行填充。

完成以上填充操作后，右面的内容将具有像素一致性，如下图。可以明显看出，其平面被放大了一圈

{% asset_img extended_face.png extended_face %}



### Implementation details

本文使用 HEVC 参考软件实现，分为两个阶段。

第一阶段，先完成当前帧的编码，若当前帧是参考帧，则计算六个面的大致纹理一致性图，得到 <a data-fancybox href="/paper/co-projection-3d-padding/example1.png" data-no-instant >Fig.5</a>，在对每个平面进行扩张得到 <a data-fancybox href="/paper/co-projection-3d-padding/extended_face.png" data-no-instant >extended_face 图</a>

第二阶段，使用扩张填充参考帧。如编码 右平面 的CU时，将右平面的扩张填入当前 CU 的每个参考帧中，得到 <a data-fancybox href="/paper/co-projection-3d-padding/padding_result.png" data-no-instant >padding_result</a>。从整张图来看，该帧并不连续，然而对于右面而言，在预定义的搜索范围 m 内，其纹理是连续的。编码完当前面平面的 CU 后，重新使用原始值填充参考帧，用于下一次编码过程中对其他面的扩张填充作准备。

{% asset_img padding_result.png padding_result %}

在参考帧之前，解码过程将用于每个 CU，我们已知当前 CU 的 MV。因此可以根据 MV 的值来判断是否需要使用当前面的扩张来填充当前 CU，以避免没必要的扩张，减少解码复杂度。



## Experiment

软件：HEVC 参考软件 HM-16.6

测试条件： RA main 10、LD main 10 和 LDP mian 10，

参数设置：quantization parameters (QP) 为 22，27，32，37; m 设置为64。

评价方式：BD-rate（Bjontegaard Delta rate）;  WS-PSNR [10] and S-PSNR [11]

对比编码方式：HEVC anchor

数据集：论文[^12] ，并且使用论文 [^3] 中的转换工具将 10 bit 的 4 x 3 立方体格式



实验结果：相对 HEVC anchor， R-D 表现中本文方法的 Y 通道在 RA、LD 和 LDP 上各有1.1%，1.2% 和 1.2% 改善。U 和 V 通道大致有 1.3%、1.5%, 和 1.3% 的比特率减少。并且具有相对最大运动的 `DrivingInCountry` 数据集上，Y 通道能到达 3.3%、3.4%, 和 3.3% 的改善。并且在所有的测试序列上，R-D 表现都有改善。且从 R-D 曲线上看，相对于 HEVC anchor，高比特率和低比特率下本文算法的 Y 通道 PSNR 均有改善。

改善效果与序列特征有关，对于在平面边界具有大范围运动的数据集，如 `DrivingInCountry`，在平面边界的 MC 会很多，由此比特率的减少会比较显著；而对于平面边界几乎不存在运动的数据集，如 `Harbor`，其 MC 发生很少，改善效果也不明显。



## Annotation

YUV - [YUV颜色编码解析 - 简书](https://www.jianshu.com/p/a91502c00fb0)

motion compensation - [运动补偿 - 维基百科，自由的百科全书](https://zh.wikipedia.org/wiki/%E8%BF%90%E5%8A%A8%E8%A1%A5%E5%81%BF)

[^Lanzcos]: [几种插值算法对比研究 - Trent1985的专栏 - CSDN博客](https://blog.csdn.net/trent1985/article/details/45150677?tdsourcetag=s_pctim_aiomsg)
[^3]:  [AHG8: InterDigital's projection format conversion tool](https://scholar.google.com.hk/scholar?cites=3770646130758356990&as_sdt=2005&sciodt=0,5)
[^12]: [JVET common test conditions and evaluation procedures for 360-degree video](https://scholar.google.com.hk/scholar?cites=15933594121870911776&as_sdt=2005&sciodt=0,5)

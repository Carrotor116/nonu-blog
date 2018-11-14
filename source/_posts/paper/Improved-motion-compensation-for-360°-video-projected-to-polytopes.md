---
title: Improved motion compensation for 360° video projected to polytopes
id: motion-compensation-optimization
category: paper
date: 2018-11-09 20:48:46
tags: ['motion compensation', '360 video']
mathjax: true
---


本文提出一种在凸面体映射表达的全景视频序列上进行正确运动补偿的方式。


<!-- more  -->

论文[^6][^7] 中提出了类似方法，不同之处在于他们都是基于线性方程直接计算其他平面上的对应点，其相应的推导仅适用于立方体映射，不能用于其他凸面体映射。而本文方法可适用于任意凸面体映射。


> 论文：[Improved motion compensation for 360° video projected to polytopes](https://ieeexplore.ieee.org/abstract/document/8019517)


---

## 360°  Video Representation

根据失真类型，全景视频的表达方式可以分为两类

1. 整张图的都存在几何失真，如 equirectangular projection, segmented sphere projection
2. 几何失真仅存在于平面边界，如 convex polytopes projection

在凸面体映射中，几何失真的程度取决于平面与邻接面的角度大小。角度小的失真严重。本文方法以立方体映射为例说明。



## Polytope Face Extension

对于某个平面，使用邻近该面的平面上的像素对其进行扩展，可得到一个更大的平面，该平面将没有主要的几何失真。以下图为例说明

{% asset_img extension.png %}



其中平面`A` 和 `B` 为邻接平面，是有针孔相机 `C` 捕获的平面。相机位于对象的中心，即相机到各个面的距离相等。由此可以将平面 `B` 上的任意点转换到平面 `A` 上。设 $K_A$ 、$K_B$ 分别为平面 `A` 和 `B` 的相机矩阵（内参）

> 相机矩阵内参  K 可以将 3D 相机坐标变换到 2D 齐次图形坐标

相机矩阵内参 K 可以表示为
$$
K = \left ( \begin{matrix}
f &  & p_x \\
 & f & p_y \\
 &  & 1\\
\end{matrix}\right )
$$
其中 $f$ 为平面到相机的距离（焦距），点 $(p_x,p_y)$ 为平面上相机的垂直投影点（主点）。然后平面 `B`　到平面 `A` 的变化如下

* 首先将平面坐标 $p_B$ 转换为相机坐标 $p_{B_{3D}}$ ，其中 $Z$ 为点 $p_{B}$ 到相机的距离

$$
p_{B_{3D}} = \left( \begin{matrix}
  & K_B^{-1} & \\
0 & 0 & Z^{-1}
\end{matrix} \right)p_B
$$

* 通过旋转平面 `B`  对应的相机坐标系，得到平面 `A`  对应的相机坐标系，将 3D 点 $P_{B_{3D}}$ 变换为 $P_{A_{3D}}$ 。其中 $R(\theta)$ 为旋转矩阵，仅与两平面夹角有关

$$
P_{A_{3D}} = \left(\begin{matrix}
R(\theta) & 0 \\
0 & 1
\end{matrix}\right)
$$

* 将 3D 点 $p_{A_{3D}}$ 变化到平面 `A` 坐标系上

$$
p_A = \left( \begin{matrix}K_A & 0\end{matrix}\right) p_{A_{3D}}
$$

综合上诉步骤，可以得到将平面坐标点 $p_B$ 映射到 平面坐标点 $p_A$ 的变换矩阵 $H_{B2A}$
$$
\begin{aligned}
H_{B2A} & = \left(\begin{matrix}K_A & 0\end{matrix}\right)
\left(\begin{matrix}R(\theta) & 0 \\
0 & 1\end{matrix}\right)
\left(\begin{matrix} & K_B^{-1} \\ 0 & 0 & Z^{-1}\end{matrix}\right) \\
& =K_A R(\theta) K_B^{-1}
\end{aligned}
$$


## Cube Face Extension

使用立方体映射为例，描述本文方法的使用。假设立方体每个面的分辨率是相同的，并且相机位于立方体的中心。即有如下相机矩阵
$$
K_A=K_B = \left( \begin{matrix}
f & 0 & 0\\
0 & f & 0\\
0 & 0 & 1
\end{matrix}\right)
$$
并且焦距 $f$ 是立方体边长的一半，$f=\frac{w}{2}$ 。假设平面 `A` 位于平面 `B` 的左边，则可以确定旋转矩阵，且得到变换矩阵 $H_{B2A}$
$$
\begin{aligned}
H_{B2A} & = K_AR_y(90^\circ)K_B^{-1}\\
& = \left(\begin{matrix}
f&0&0\\0&f&0\\0&0&1
\end{matrix}\right)
\left(\begin{matrix}
0&0&1\\0&1&0\\-1&0&0
\end{matrix}\right)
\left(\begin{matrix}
f^{-1}&0&0\\0&f^{-1}&0\\0&0&1
\end{matrix}\right)\\
&=f^{-1}
\left(\begin{matrix}
0&0&f^2\\0&f&0\\-1&0&0
\end{matrix}\right)
\end{aligned}
$$

因为齐次坐标仅按比例定义，可以消除缩放因子 $f^{-1}$  。经过扩展操作可以得到如下效果图。可见本方法可以矫平面像边缘的几何一致性。**但是随着里平面边缘距离的增大，采样密度将减小，产生像素失真情况** 。

{% asset_img result.png %}


可以相应地得到其他邻接面的映射，完成该平面四个方向的扩展。



## Integration into Coding Scheme

本文假设输入的全景视频序列为 3 x 2 的紧凑型立体展开布局。平面扩展仅用于运动补偿时对参考帧的修改，对参考帧序列的每个参考帧都进行扩展。对每个平面都生成额外六个参考帧。因为两张邻接图的扩展范围会存在覆盖重叠，所以需要生成 6 个扩展的参考帧。对应每个平面的扩展参考帧如下

{% asset_img reference.png %}


其中扩展的范围 $N$ 是可以配置的，本文设置为与运动补偿相同的值。

**基于全景视频的对称性：超出某平面的内容将会出现在了一个平面的扩展区域中**。扩展参考帧中的剩余区域（上图白色区域）不保留数据，以便于简化实现。若通过仅存储扩展区域，可以进一步提高存储效率。

在测试序列中，平面的宽高均为 8 的倍数，这也是 HEVC 中 CU 的最小大小，因此每个 CU 均会位于一个平面上。又因为在编解码过程中 CU 的位置是已知的，所以对于给定 CU ，其所需要的扩展参考帧可以通过 CU 本身的位置推导得到。即本算法不需要额外的信息。



## Experiment

本文算法实现基于 HEVC 参考软件 HM 16.7 [^13] ，其中映射实现使用了 OpenVC [^14] 。在扩展中的像素值计算使用了线性插值法。本实现中将一个参考帧替换为了 6 个参考帧。

**数据集：**GoPro [^15] 中的五个序列，另外 9 个序列取自 CTC360 [^16] 。其中一半为静态相机序列，一般为非静态相机序列。

**编码配置：**RA配置，且 $QP\in \{22,27,32,37\}$  

**对比算法：**HEVC 参考软件 HM 16.7

**评价标准：**Bjontegaard Delta (BD)

**实验结果：**

1. 静态序列中，平均 BD 改善 0.06%；

2. 非静态序列中，平均 BD 改善 2.14%，且 PSNR 平均提升 0.07 dB。

   本方法用于运动补偿，故适用于具有大量全局运动的序列。

3. 在少数序列中，存在负面效果。即原始参考图上的平面边界预测效果比扩展参考图的预测效果好。

4. 内存消耗上是 HM 16.7 的七倍。因为本文仅采取了简单实现。

   实际中真需要的额外存储空间 $E_{Ext}$ 为 $A_{Ext}=6(4NW - 4N^2)$ 。由于 $N$ 远小于平面的宽度 $W$ ，所以额外开销很小。没有填充的真面积就为 $6W^2$ 。

5. 额外计算成本比较小，每个参考帧仅需要一次扩展计算。且运动搜算的计算量没有变化。



## Annotation

[^6]: [Co-projection-plane based 3-D padding for polyhedron projection for 360-degree video](https://ieeexplore.ieee.org/abstract/document/8019393/)
[^7]: Yuwen He, Yan Ye, Philippe Hanhart, et al., “Geometry padding for 360 video coding,” Doc. JVET-D0075, Joint Video Exploration Team (on Future Video coding) of ITU-T VCEG and ISO/IEC MPEG, Chengdu, CN, 4th meeting, Oct. 2016.
[^13]: [svn_HEVCSoftware - Revision 4995: /tags/HM-16.7](https://hevc.hhi.fraunhofer.de/svn/svn_HEVCSoftware/tags/HM-16.7/)
[^14]: [GitHub - opencv/opencv: Open Source Computer Vision Library](https://github.com/opencv/opencv)
[^15]: Adeel Abbas, “GoPro test sequences for virtual reality video coding,” Doc. JVET-C0021, Joint Video Exploration Team (on Future Video coding) of ITU-T VCEG and ISO/IEC MPEG, Geneva, CH, 3rd meeting, May2016.
[^16]: [JVET-G1030: JVET common test conditions and evaluation procedures for 360° video](https://www.researchgate.net/publication/326504378_JVET-G1030_JVET_common_test_conditions_and_evaluation_procedures_for_360_video)

---
title: Stereo Seam Carving a Geometrically Consistent Approach
id: stero_same_carving
date: 2018-10-17 15:50:06
tags:
mathjax: true
---


本文提出将 seam carving 用于立体图像的方法，以实现在缩放图像时能够维持原来的几何一致性。

本文中的图像缩放，是指仅仅对宽度或高度进行缩放，所以图像缩放后，其3D结构中存在几何一致性问题。若按比例**均匀**缩放宽高，则几何一致性不会发生改变。
<!--more-->

> 论文:  [Stereo Seam Carving a Geometrically Consistent Approach](https://ieeexplore.ieee.org/document/6468042)



---



## BACKGROUND

根据如何决定图片中不同像素的重要程度，以及如何利用重要程度这个信息，image and video retargeting algorithms 可以分为两主要的类：

1. 删除或转移像素的用于单图像重定目标的**离散方法**， 如seam carving [^3]、shift map [^6]
2. 基于图像内容进行扭曲四边形网格的**连续方法**，如 [^7]、[^8]



seam carving algorithm: 迭代地进行计算具有最小视觉失真的 seams 并移除或放大该 seams.

一种利用 seam carving 来维护3D信息的方式：计算左图的 seam，然后通过 视差图[^视差图] 将 seam 映射到右图中（不足之处是未利用右图或者 深度图[^深度图] 的信息）。

本文采用的 seam carving 的方式：


> simultaneously carve a pair of seams in both images while minimizing distortion in appearance and depth.

> Seam selection and disparity map modification are subject to geometric constraints that take into account the visibility relations between pixels in the images (occluded and occluding pixels).





## THE METHOD

输入：一对 $m\times n$ 的整流立体图片 $\{I\_{L}, I\_{R}\}$， 和视差图 $D$ (本文使用 [^5] 计算得到 )

输出：$\{\hat{I\_{L}}, \hat{I\_{R}}\}$ 和 更新后的视差图 $\hat{D}$

参考图：左图像的视差作为参考图

目标：得到与可行3D场景的几何一致的调整过大小的图片



constraints:

1. 原始图像中对应的像素要么都被移除，要么都被保留
2. 在参考图中可见，但另一个中不可见的 3D 点不显示



key :

1. 定义 seam coupling
2. 定义 energy function 用于选取 seams
3. 在3D图像上维持像素可见性



### Seam coupling 

使用由 $D$ 定义的对应关系 来获得几何对偶的两个 seams $S\_{L}=\{s\_{l}^{i}\}\_{i=1}^{m}$ 和 $S\_{R}=\{s\_{r}^{i}\}\_{i=1}^{m}$
$s\_{l}^{i} \in S\_{L}$ 代表左图像的 seam 中第 $i$ 行的 pixels， $s\_{r}^{i} \in S\_{R}$ 代表右图像的 seam 中第 $i$ 行的 pixels ，这些像素相互匹配。
$$
s_{R}^{i}=(i,j_{R}(i)) = (i,j_{L}(i)+D(s_{L}^{i}))
$$
其中 $j\_{l}, j\_{R}: [m]\to [n]$ 且 $[m] = [1, .., m]$ 。如果像素的相对关系存在，估计的视差图 $D:[n]\times [m]\to Z \cup\perp$ 将 $I\_{L}$ 的像素映射到 $I\_{R}$ 的像素上，否则映射到 $\perp$ 。seams 仅含有定义了视差的像素。

左图像中的连续 seam 通常对应右图像的分段连续 seam，因为 seam 可能会穿过深度不连续的点。所以我们假设 seam 是分段连续的，并且称之为广义接缝（generalized seams）

 

### The Energy Function

能量函数由两部分组成

* `强度项`，即左右图片上平面像素的能量：**x、y轴**，为 [Appearance Energy](#appearance-energy)
*  `3D 几何项`，即左右图像的视差信息的能量：**z轴**，为 [Depth Energy](#depth-energy)



重新定位的左右图像中的产生的梯度（result gradients）分别依赖于先前列（previous row）上的 seam 像素（该像素记为 $j^{\pm}\_{L}$ 和 $j^{\pm}\_{R}$）。由于 seam 是对偶的，所以 $j\_{R}^{\pm}$ 由 $j\_{L}^{\pm}$ 和视差图 $D$ 唯一确定。因此依据 **先前行** 上的 seam 像素 $j^{\pm}$ （$j\_{L}^{\pm}$ 的缩写） 来定义能量函数
$$
E_{total}(i,j,j^{\pm}) = E_{intensity}(i,j,j^{\pm}）+\alpha E_{3D}(i,j,j^{\pm})
$$
由于我们使用广义接缝，所以 $j^{\pm} \in [m]$ （而在连续接缝中，$j^{\pm}\in \{j-1,j,j+1\}$）



#### Appearance Energy

Forward energy [^4] ：其目标在于最小化（由新产生的邻居像素之间的强度差异所引起的）重定向图的结果失真。

外观失真 $E\_{intensity}(i,j,j^{\pm})$ 被认为是（由移除左右图像中的对偶像素所引起的）能量 $E\_{L}$ 和 $E\_{R}$ 的总和
$$
E_{intensity}(i,j,j^{\pm}) =E_{L}(i,j,j^{\pm}) + E_{R}(i,j_{R},j^{\pm}_{R})
$$
其中通过视差图来捕获左右 seams 的耦合

移除图像 $I$ （左图像或右图像）上一个特定像素 $(i,j)$ 的能量如下
$$
E(i,j,j^{\pm})=E^{v}(i,j,j^{\pm})+E^{h}(i,j)
$$
其中 $E^{h}$ 和 $E^{v}$ 分别是是由于垂直和水平方向上新的梯度产生的向前能量项（forward energy terms）
$$
E^{h}(i,j) = |I(i,j+1)-I(i,j-1)|
$$
{% asset_img E^h.png [E^h 计算] %}

在垂直方向上，新梯度依赖于第  $i-1$ 行中 seam 的的位置 $j^{\pm}$
$$
E^{v}(i,j,j^{\pm}) = \begin{cases}
\begin{aligned}
& V_{1} & j^{\pm}< j \\
& 0 & j^{\pm}=j \\
& V_{2} & j^{\pm}>j 
\end{aligned}
\end{cases}
$$
其中
$$
\begin{aligned}
& V_{1} = \sum_{k=j^{\pm}+1}^{j}|I(i-1,k)-I(i,k-1)|\\
& V_{2} = \sum_{k=j+1}^{j^{\pm}}|I(i-1,k-1)-I(i,k)|
\end{aligned}
$$

{% asset_img E^v.png [E^v 计算,(j ± < j)] %}



#### Depth Energy

计算得到的深度图提供了 seam 选择的有价值的线索，3D向前能量项 $E\_{D}$ 用于最小化视差失真。$E\_{D}$ 的定义与 $E\_{intensity}$ 很类似，通过使用视差图 $D$ 替换 $E\_{intensity}$ 中的强度函数 $I$ 实现。

为了弥补强度（intensity）和视差（disparity）值的差异，将 $I$ 和 $D$ 都归一化到 0 至 1 的范围。

我们偏向于移除具有高置信度视差值的像素，其视差值由对应像素强度的差异来衡量
$$
G(i,j) =|I_{L}(i,j) - I_{R}(i,j+D(i,j))|
$$
总的向前3D能量由三部分的加权和组成
$$
E_{3D} (i,j,j^{\pm}) = E_{D}(i,j,j^{\pm})+\beta|D_{n}(i,j)|+\gamma G(i,j)
$$
其中 $D\_{n}$ 是标准化的视差图



### Maintaining Pixel Visibility

constraint 2 中提到”在参考图中可见，但另一个中不可见的 3D 点不显示“，因为否则将没有连贯的3D解释可以证明 ”（只在一个图像中可见，在另一个图像中不可见的）显示像素 (revealed pixel) 的可见性“

通过”避免删除遮挡像素“来保证”（原右图像中已被遮挡的）像素在重定向右图像中依然被遮挡“

本文对删除像素的选择既不是遮挡像素，也不是被遮挡像素，这可以保证**维持原可见性的关系**。

遮挡像素和被遮挡像素使用视差图 $D$ 计算并保存在 $O(i,j)$ 中，$O(i,j)=1$ 表示 $(i,j)$ 像素为遮挡或被遮挡像素，使用 `simplified Z-buffer approach` 计算。



### Stereo Seam Selection and Carving

计算能量作为 cost matrix $M$，当 $O(i,j)=1$ 时记 $M(i,j)=\infty$ 

选择 seams 时，默认选择连续的 seams，因为其 appearance energy 影响较少（根据公式 $V\_{1}$、$V\_{2}$），当由遮挡像素引起阻断时，允许选择非连续 seams
$$
M(i,j)=\begin{cases}
\begin{aligned}
& \min \limits_{j^{\pm}\in\{j-1,j,j+1\}} E_{total}(i,j,j^{\pm});& T(i,j)=0 \\
& \min \limits_{j^{\pm}\in[m]} E_{total}(i,j,j^{\pm});& T(i,j)=1
\end{aligned}
\end{cases}
$$
$T$ 是二值图，$T(i,j)$ 表示连续路径是否被第 $i-1$ 行的像素遮挡（或被遮挡）
$$
\begin{aligned}
& T(i,j) = \begin{cases}
1; & O(i-1,j^{\pm}) = 1\\
0; & O(i-1,j^{\pm}) = 0
\end{cases}\\
& j^{\pm} \in \{j-1, j, j+1\}
\end{aligned}
$$
移除 seams 将造成像素漂移，作图的漂移函数 $f\_{L}(i,j):[m]\times[n]\to [m]\times[n-1]$ 。（因为 seams 为纵向，所以产生seams 右侧像素的左漂）。定义 $s\_{l}^{i} = (i, j\_{L}(i))$ 为左图中被移除的像素，则有
$$
f_{L}(i,j)=\begin{cases}\begin{aligned}
&j & if \quad j< j_{L}(i)\\
&j-1 & if \quad j> j_{L}(i)\\
&\perp & if \quad j=j_{L}(i)\\

\end{aligned}\end{cases}
$$
右图漂移函数同理可得。

雕刻接缝后将产生新的视差图 $\hat{D}$，通过移除 $D$ 上的左 seam $S\_{L}$ 再更新剩余像素的视差来获得
$$
\hat{D}(i, f_{L}(i,j)) = f_{R}(i, j+D(i,j)) - f_{L}(i,j)
$$
$(i,j+D{(i,j)})$ 表示雕刻接缝前左图中 $(i,j)$ 像素对应的右图像素坐标

#### Geometric Interpretation

1. 当左右图像中对应两个像素未发生漂移，其关联的 3D 点保持不变
2. 当左右图像中对应两个像素都发生漂移，其关联的 3D 点位置平行于图像平面发生左漂，其原始视差不变 $\hat{D}(i,j)=D(i,j)$
3. 当左右图像中对应的两个像素只有一个发生漂移，视差图将根据其中一个像素发生改变，这对应一个小的深度变化。

{% asset_img d2.png [左图像中一像素发生左漂，关联 3D 点位置从Pa变化到Pb。注意只有非遮挡像素(红色)可能被选为 seams 像素] %}


### Stereo Image Pair Enlarging

放大图片宽度：

1. 和移除 seams 操作相同地选取一对 seams，将 seams 复制扩大一倍
2. 如果需要，更新视差图



## HORIZONTAL SEAMS

为了维持几何一致性，需要对视差图进行限制，选取成对的 seams。

在垂直场景下，（右图像对左图像的）像素映射集合中通常不存在对应的垂直 seams。

**在需要保持几何一致性的情况下，垂直 seams 不能用于重定向图片**



## GEOMETRIC CONSISTENCY

### 定义遮挡和被遮挡像素

左图像上两个像素 $(i,j\_{b})$ 和$(i,j\_{f})$ ，其中 $(i,j\_{b})$ 被 $(i,j\_{f})$ 遮挡，则有如下关系
$$
\begin{aligned}
& j_b < j_f\\
& j_f + D(i,j_f) = j_b+D(i,j_b)
\end{aligned}
$$
即 被遮挡像素 位于 遮挡像素的左边，且这两个像素对映射到右图像时为同一个像素。

故若 $(i,j)$ 非遮挡或被遮挡像素，则有：
$$
j+D(i,j) \neq j' + D(i,j') \quad \forall j' \neq j
$$

### 引理

 Lemma.1 同一行像素在漂移操作后它们的位置顺序不改变
$$
\begin{aligned}
j_1< j_2 \Leftrightarrow f_L(i,j_1) < f_L(i,j_2)\\
j_1=j_2 \Leftrightarrow f_L(i,j_1) = f_L(i,j_2)
\end{aligned}
$$


### 证明

假设 $p\_f(i,j\_f)$ 和 $p\_b(i,j\_b)$ 是参考图像中两个像素，且 $p\_f$ 遮挡了 $p\_b$。求证 $(i,f\_l(i,j\_f))$ 遮挡了 $(i,f\_l(i,j\_b))$ ，即移除 seams 后对应的漂移像素仍然保持原来的遮挡关系

证：

原题即使证明 $f\_l(i,j\_b)< f\_l(i,j\_f)$ 且 $f\_l(i,j\_B)+\hat{D}(f\_l(i,j\_b)) = f\_l(i,j\_f) +\hat{D}(f\_l(i,j\_f))$
$$
\begin{aligned}
& f_L(i,j_b)+\hat{D}(f_L(i,j_b))\\
& \quad = f_L(i,j_b) + f_R(i,j_b+D(i,j_b)) - f_L(i,j_b) \\
& \quad  = f_R(i,j_b+D(i,j_b))\\\\

& f_L(i,j_f)+\hat{D}(f_L(i,j_f))\\
& \quad = f_L(i,j_f) + f_R(i,j_f+D(i,j_f)) - f_L(i,j_f)\\
& \quad = f_R(i,j_f+D(i,j_f))\\\\

& j_b< j_f \\
& \quad \Rightarrow f_L(i,j_b)< f_L(i,j_f)\\\\

& j_b+D(i,j_b) = j_f+D(i,j_f) \\
& \quad \Rightarrow f_R(i,j_b+D(i,j_b)) = f_R(i,j_f+D(i,j_f))\\
& \quad \Rightarrow f_L(i,j_b)+\hat{D}(f_L(i,j_b)) =  f_L(i,j_f)+\hat{D}(f_L(i,j_f))
\end{aligned}
$$



**表示移除 seams 产生的像素漂移操作后，3D点的可见性能与操作前保持一致**
 


## EXPERIMENTS AND RESULTS

视差图：computed by the SGM stereo algorithm [^5] and hole filling

数据集：

1. Middlebury，六类（Moebius , Aloe , Cloth, Wood, Dolls, Laundry），具有复杂纹理，包含不同深度的对象约20% 的相似属于遮挡或被遮挡像素
2. Portrait，大部分图像覆盖着的显著性对象，需要保证显著性对象不失真。且具有大部分区域只存在于左图像而不存在于右图像
3. Flickr，其图像的深度范围很大

### 几何一致性评估

评估视差失真的百分比 $B$（测量原始视差图 $\hat{D}\_o$ 和更新视差图 $\hat{D}\_{SGM}$ 之间的偏差）。其中 $\hat{D}$ 由输入视差 $D$ 移除相关 seams 得到。
$$
B = \frac{1}{N}\sum_{(i,j)}(|\hat{D}_o(i,j)-\hat{D}_{SGM}(i,j)| > 1)
$$
本应该计算 **3D 失真**，但是由于相机为校准且逆视差的单位未知，所以改用计算 **视差失真** 来衡量。



### 实验参数

$\beta=0.08$ ，$\gamma=0.5$ ，$\alpha$ 从1到5。Aloe 和 Moebuis 缩小率 20%，其他数据集缩小率 17% 。

对比对象：

1. 简单独立的图片重定向（naive independent retargeting of each image）
2. 简单视差图映射使用来选择另一个图像的seams （a naive usge of the disparity to retarget the reference image and map the selected seams to the other image）
3. 基于变形 ("wraping-based") 的立体图像重定向方法



### 结论

1. 不同缩放率下，平均 99% 的像素深度值能够被保留

2. 使用ground truth disparity map时，几乎不产生失真

3. 与（基于变形 ("wraping-based") 的立体图像重定向方法）[^12]、[^13] 的视差图进行对比，前者视差图具有噪声和明显的人工痕迹，后者虽然噪声少，但是其3D结构和深度与输入视差图的情况不一致（深度变更深或浅。而这些3D失真和噪声不存在与本文方法的结果中

4. 红绿图的深度知觉测试中，本文方法深度与原始输入相似，而独立图像重定向（independent image retargeting）有明显失真



## CONCLUSIONS AND FUTURE WORK

提出了将seam carving 有效地用于3D图像的方法（利用表面"x,y"和深度"z"信息），该图像重定向方法能够保持图像的几何一致性。

局限性：

1. 本文方法效果主要受限于 disparity map 的输入和遮挡（被遮挡）像素的数量。两者都受纹理，相机位置和3D场景的影响。
2. 只能用于整流的（rectified）立体图像，并且只能作用于垂直方向的 seams。
3. 进行减小图像高度操作，则需要先扩大图像宽度再使用等比例同时调整的图像的宽高，得到缩小高度的效果（uniform resizing of the image）

Future 是扩展方法到立体视频上，结合基于深度的显著性图，在配有立体摄像头和3D显示功能的智能手机上实现。另一个方向是建立一个立体重定向和编辑的基准（benchmark），进行深度感知的评估。


 
## Reference

[^视差图]: [一点点关于Disparity Map的理解 – Tommy's WordPress](https://tjbwyk.wordpress.com/2016/02/03/%E4%B8%80%E7%82%B9%E7%82%B9%E5%85%B3%E4%BA%8Edisparity-map%E7%9A%84%E7%90%86%E8%A7%A3/)

[^深度图]: [Depth map - Wikipedia](https://en.wikipedia.org/wiki/Depth_map)


[^3]: [Seam carving for content-aware image resizing](https://dl.acm.org/citation.cfm?id=1276390)
[^6]: [Shift-map image editing](https://ieeexplore.ieee.org/abstract/document/5459159)
[^7]: [Non-homogeneous Content-driven Video-retargeting](https://www.computer.org/csdl/proceedings/iccv/2007/1630/00/04409010-abs.html)
[^8]: [Optimized scale-and-stretch for image resizing](https://dl.acm.org/citation.cfm?id=1409071)
[^5]: [Stereo Processing by Semiglobal Matching and Mutual Information](https://ieeexplore.ieee.org/abstract/document/4359315)
[^12]: [Content-Aware Display Adaptation and Interactive Editing for Stereoscopic Images](https://ieeexplore.ieee.org/abstract/document/5715885)
[^13]: [Scene warping: Layer-based stereoscopic image resizing](https://ieeexplore.ieee.org/abstract/document/6247657)


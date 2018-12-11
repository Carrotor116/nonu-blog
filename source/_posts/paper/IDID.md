---
title: Interpolation-Dependent Image Downsampling
id: IDID
category: paper
date: 2018-12-09 23:15:23
tags: ['downsampling', 'upsampling', 'interpolation']
mathjax: true
---


本文提出 与插值相关的 下采样方法，以最小化输入图像和某种插值方法生成的图像之间的差异。最优的下采样值，能最小化其邻接原始像素和插值像素之间的方差之和。



<!-- more -->

金字塔条件 [^17] [^18] ： 上采样后的下采样操作应该具有身份运算符 ( identify operator)。 可是下采样后的上采样通常不具有 identify  operator。

> In [17] and [18], Goutsias and Heijmans proposed a pyramid condition that the downsampling operator after upsampling should give the identity operator. However, as stated in [17] and [18] the upsampling operator, after downsampling, usually cannot result in the identity operator.

输入信号和输出信号 ( detail signal，由上采样操作生产) 之间的**差异**对于完美重构是不可缺少的。这需要最小化输出信号的能量。而论文 [^17] [^18] 没有给出最小化 输出信号 的方法。


> 论文：[Interpolation-Dependent Image Downsampling - IEEE Journals & Magazine](https://ieeexplore.ieee.org/abstract/document/5782987)


---


## IDID

首先简述 IDID 的理论实现。


{% asset_img down_up_sample.png %}


$\rm Y$ 表示原始图像，大小 $M\times N$，$\rm X$ 表示采样后图像，大小 $M/2\times N/2$，$\hat{\rm Y} $ 为插值后图像，其中灰色点为插值像素。IDID 目标是得到能够使得插值后图像质量最高的下采样图像 $\rm X$ 。
$$
\rm X = \arg \min_{\rm X}||\hat{\rm Y}- \rm Y||^2
$$
其中 
$$
\begin{aligned} & \rm X = \rm{flat}(\begin{bmatrix}
X_{0,0},&X_{0,1},&\cdots &X_{0,\frac{N}{2}-1}\\
X_{1,0},&X_{1,1},&\cdots &X_{1,\frac{N}{2}-1}\\
\vdots & \vdots&\ddots & \vdots\\
X_{\frac{M}{2}-1,0},&X_{\frac{M}{2}-1,1},&\cdots &X_{\frac{M}{2}-1,\frac{N}{2}-1}\\
\end{bmatrix})^T\\ \quad \\
& \rm Y = \rm{flat}( \begin{bmatrix}
Y_{0,0},&Y_{0,1},&\cdots &Y_{0,N-1}\\ 
Y_{1,0},&Y_{1,1},&\cdots &Y_{1,N-1}\\
\vdots & \vdots&\ddots & \vdots\\
Y_{M-1,0},&Y_{M-1,1},&\cdots &Y_{M-1,N-1}\\
\end{bmatrix})^T
\end{aligned}
$$

给定图像 $\rm X$ ，其插值过程可以表示为
$$
\begin{gather*}\hat{\rm Y} =\rm H \rm X\\
\quad \\
\rm H = \begin{bmatrix}
h_{0,0}&h_{0,1}&\dots &h_{0,M/2\times N/2-1}\\
h_{1,0}&h_{1,1}&\dots &h_{1,M/2\times N/2-1}\\
h_{2,0}&h_{2,1}&\dots &h_{2,M/2\times N/2-1}\\
\dots &\dots &\dots &\dots \\
h_{M\times N-1,0}&h_{M\times N-1,1}&\dots &h_{M\times N-1,M/2\times N/2-1}\\
\end{bmatrix}
\end{gather*}
$$
得到目标函数 $J$
$$
J =\min_{\rm}||\rm H\rm X-\rm Y||^2
$$
令其偏导数为 0 
$$
\frac{\partial J}{\partial \rm X} =2\rm H^T(\rm H\rm X-\rm Y)= 0
$$
解得最优下采样 $\rm X^\star$ 
$$
\rm X^\star = (\rm H^T \rm H)^{-1}\rm H^T \rm Y
$$

> 显然该最优下采样与插值矩阵 $\rm H$ 相关



## Content-dependent IDID

IDID 在进行下采样时需要已知插值矩阵 $\rm H$ 。因此对于内容无关的插值方法，可以直接使用 IDID；而对于内容相关的插值方法，由于下采样前无法获得插值矩阵，而不能直接使用 IDID 。故提出 Content-dependent IDID

伪码：

```
initialze H^0 and X^0
for i=1 to iMax-1
  Compute H^i based on X^{i-1} according to interpolation method
  Compute X^i according to IDID
  Compute the MSE: E(i)=4/(M*M) *||X^i - X^{i-1}||^2
  if E(i) < T
     break
  end if
end for
```

1. 初始化 $\rm H^0$ 和 $\rm X^0$，其中 $\rm H^0$ 由双线性插值器构成，$\rm X^0$ 由直接下采样获得
2. 基于下采样结果 $\rm X^{i-1}$ ，使用基于内容的插值方法，计算插值矩阵 $\rm H^i$
3. 基于插值矩阵 $\rm H^i$，使用 IDID 公式 ( 计算 $\rm X^\star$ 的公式) 求得 $\rm X^i$
4. 计算 $\rm X^{i-1}$ 和 $\rm X^i$ 之间的差异 $\rm E(i)$
5. 若 $\rm E(i)$ 小于阈值或达到迭代次数上限则停止迭代，否则从 2 开始迭代

本文设置阈值 $\rm T$ 为 0.5


{% asset_img p2.png %}


上图为迭代次数与插值图像的 PSNR 的关系。IDID_EDI+EDI 表示首先对输入图像进行 IDID 下采样，其中插值矩阵由 EDI 相关系数组成，之后使用 EDI 对进行迭代生成下采样图像。EDI 与 NLEDI 之间的差异为，根据每个采样像素与中心采样像素的结构相似度，NLEDI 会为每个采用因素分配一个独一无二的权值。IDID_EDI 和 IDID_NLEDI 能够生成具有高 PSNR 插值图像的下采样图像，且迭代次数为 2 时，PSNR 达到收敛状态。



> 迭代为什么会收敛？
>
> 将单个的基于内容的插值操作表示为
> $$
> \begin{aligned} &\rm H = F(X)\\ &\rm Y' = HX\end{aligned}
> $$
> 则上述迭代操作可以表示为
> $$
> \begin{aligned}
> & \rm H^i  = F(X^{i-1})\\
> & \rm X^i = IDID(H^i) \\ \quad \\
> & \rm X^i = IDID(F(X^{i-1}))
> \end{aligned}
> $$
> 根据 IDID 性质，给定 $\rm H^i$ 下，$\rm X^i H^i$ 具有最优插值效果
> $$
> \begin{aligned}
> &\rm Q(X^{i-1} H^i) \le Q(X^iH^i) \le Q(Y) 
> \end{aligned}
> $$
>
> 可是随着迭代进行，$\rm H^j$ 发生改变，为什么 $\rm X^i$  会比 $\rm X^{i-1}$ 要好？




## Experiment

### Blockwise implementation

由于矩阵 $\rm H$ 维度 $MN \times MN/4$ 太大，而提出逐块 IDID 。

{% asset_img blockwise-IDID.png %}


块采用过程如上图，其中实线表示一个块，白色点为下采样像素，黑色圆点为对角线方向的插值像素，黑色方块为水平方向的插值像素，三角黑块为垂直方向的插值像素。快边缘的插值像素会涉及到块外的下采样点，如灰色插值像素。插值 $\hat{\rm Y}$ 可以表示为
$$
\hat{\rm Y} =\rm H \rm X + \Phi
$$
其中 $\Phi$ 为一个列向量，表示块外下采样像素的贡献。对于不涉及块外采样像素的插值，$\Phi$ 为 0 ，对于有涉及到块外采样像素的插值，$\Phi$ 的对应元素为插值系数与当前块外部的对应像素的乘积之和。

实验中 IDID 块大小为 16 。



### Downsampling and Interpolation Comparisons 

在七类不同大小和内容的数据集（ Boat (512 512), Lena (512 512), Elaine (512 512), Couple Parrot (768 512), 和 Motor (768 512) ）上，进行 1/4 倍数的下采样，再插值还原回原始大小。

插值方式：四种， Bilinear 、Bicubic、 EDI 和 NLEDI 。

下采样方式：六种

1. 直接下采样，直接取四个相应像素值中左上角的值 
2. MPEG-B 采样，先对图像进行过滤以减少带宽，再使用 (1) 直接采样
3. IDID_Bilinear，使用 IDID 下采样，其中插值矩阵由 Bilinear 插值系数组成
4. 与 (3) 类似的 IDID_Bicubic 、IDID_EDI 和 IDID_NLEDI

通过计算 PSNR ，得到三个结果

1. 下采样方法对插值图像具有很大的影响。直接采样和 MPEG-B 采样的 PSNR 最低
2. IDID 下采样能保存更多的图像信息，得到更高的 PSNR
3. 对于每种插值方式，使用对应插值相关系数的 IDID 下采样方式能达到最大 PSNR

上诉三个结果进一步验证了 IDID 的有效性。

且从下采样图像的效果看，其不仅提高了插值质量，也提高了下采样图像的视觉效果



### Low-Bit-Rate Image Compression

基于 IDID 的低比特率图像压缩，包含四个组建 IDID 、JEPG Encoder、JEPG Decoder 和 Interpolation


{% asset_img IDID-IC.png %}


各个压缩方式在不同比特率上的 PSNR 如下

{% asset_img p3.png %}


> 图像的比特率为位深，表示每个颜色通道的位数，单位 bpp (bits per pixel)。
>
> 个人理解，低比特率压缩是对图像的位深进行压缩

根据结果可知，

1. 直接下采样-插值 的压缩结果比无下采样的 JEPG 压缩的好，因为只有 四分之一 的原始数据需要被压缩

   > 这句话忽视了 下采样-插值 导致的质量损失

2. 基于 IDID 的压缩方法效果比其他方法要好，因为 IDID 能够在下采样中保存更多信息

从图像视觉效果看，

1. JEPG 压缩结果具有严重伪痕
2. Direct-NLEDI (直接下采样和NLEDI插值) 压缩结果没有伪痕，但是存在很大噪声。因为直接下采样导致大量信息丢失
3. IDID_Bilinear-Bilinear 和 IDID_NLEDI-NLEDI 表现最优 





## Annotation 



EDI : [Edge-Directed Interpolation](http://chiranjivi.tripod.com/EDITut.html)

[^17]: [Nonlinear multiresolution signal decomposition schemes. I. Morphological pyramids](https://ieeexplore.ieee.org/abstract/document/877209)
[^18]: [Nonlinear multiresolution signal decomposition schemes. II. Morphological wavelets](https://ieeexplore.ieee.org/abstract/document/877211)

---
title: Non-local Neural Networks
id: non-local-neural-networks
category: paper
date: 2019-01-19 13:45:41
tags: ['non-mean']
mathjax: true
---


本文提出 non-local operation 用于在深度神经网络中高效地提取长范围依赖。

<!-- more -->

具有一下优势：

1. 直接计算长距离依赖，不受距离范围限制
2. 计算高效
3. 输入变量的维度不受限，故容易和网络中的其他操作连接



---



## Non-local Neural Networks

non-local operation 与 non-local mean 有相似的地方。

non-local mean 是一种滤波算法。
一般的局部滤波器会考虑目标像素邻近的像素，如取邻近像素的均值更新目标像素的值，这属于一种 local 的算法。而 non-local mean 的不同之处在于，它会考虑整个图像上所有的像素来更新目标像素。根据图像所有像素与目标像素的相似度，给予一个相似度权重，将所有像素按相似读权重取均值，来更新目标像素。数学表示如下
$$
NL({\rm I}_i) = \sum_{\forall j} W({\rm I}_i,{\rm I}_j) \cdot {\rm I}_j
$$
其中 $i$ 和 $j$ 为图像中像素位置索引。因为 $j$ 遍历了所有位置，所以是一种 non-local 的算法。




### Formulation

non-local operation 的数学表示为
$$
{\rm y}_i=\frac{1}{C({\rm x})}\sum_{\forall j}f({\rm x}_i,{\rm x}_j)g({\rm x}_j)
$$

$i$ 和 $j$ 是位置索引，可以是时间坐标、空间坐标或时空坐标上的索引；${\rm x}$ 和 ${\rm y}$ 为输入信号和输出信号，二元函数 $f$ 计算输入输出信号对应索引位置的表示之间的关系（标量），一元函数 $g$ 计算输入信号在索引 $i$ 处的表示；$C({\rm x})$ 是标准化因子；$\forall j$ 表示对所有位置的表示进行计算，即 non-local 。

相对于 non-local ，卷积 (convolutional) 操作是对空间上 local 信息进行求和，循环 (recurrent) 操作是在时间的 local 范围（当前时刻和上一个时刻）上进行操作。

对比与全连接层，non-local 使用<u>不同位置上的关系</u>计算，而 full-connected 使用<u>学习到的权重</u>计算；且non-local 的输入输出大小相同，而 full-connected 可能不同；full-connected 一般只放在模型的最后几层，而 non-local 可以放在深度网络的任意位置。



### Instantiations

讨论不同的 $f$ 和 $g$ 函数的效果。为了简化，取线性 embedding  $g({\rm x}\_j)=W\_g{\rm x}\_j$  其中 $W\_g$ 为权重矩阵，由卷积学习得到。对 $f$ 讨论如下形式时的效果



**Gaussian**

类似与 non-local mean [^4] ，可以定义 $f$ 为高斯函数
$$
f({\rm x}_i,{\rm x}_j) = e^{ {\rm x}_i^T {\rm x}_j}
$$
${\rm x}\_i^T {\rm x}\_j$ 为点乘相似度，也可以计算欧式距离，不过矩阵乘积更好实现。此时标准化因子为 $C({\rm x})=\sum\_{\forall j}f({\rm x}\_i,{\rm x}\_j)$



**Embedding Gaussian**
$$
\begin{gather}f({\rm x}_i,{\rm x}_j) = e^{\theta({\rm x}_i)^T \phi ({\rm x}_j)}\\ \quad\\
\theta({\rm x}_i) =W_{\theta} {\rm x}_i\\
\phi({\rm x}_j) = W_{\phi} {\rm x}_j\\
C({\rm x})=\sum_{\forall j}f({\rm x}_i,{\rm x}_j)
\end{gather}
$$
这是 Gaussian 的一种简单扩展。

对于给定 $i$ ， $\frac{1}{C({\rm x})}f({\rm x}\_i,{\rm x}\_j)$ 是一个 softmax 函数，所以有
$$
{\rm y} = softmax({\rm x}^TW_{\theta}W_{\phi}{\rm x})g({\rm x})
$$
其中 $j$ 是对所有位置的遍历，所以<u>这是一种 self-attention 形式</u> [^49]。



**Dot product**

定义 $f$ 为点乘相似度
$$
f({\rm x}_i,{\rm x}_j) = \theta({\rm x}_i)^T\phi({\rm x}_j)
$$
这种情况下，标准化因子定义为 $C({\rm x})=N$ ，其中 $N$ 是 ${\rm x}$ 的位置数量，这样能简化梯度的计算。

> A normalization like this is necessary because the input can have variable size.

点乘和 embedding Guassian 的差异在于后者是 softmax 的一种表现。



**Concatenation**

这是在 Relation Network 中使用的一种二元函数
$$
f({\rm x}_i, {\rm x}_j)={\rm ReLU}({\rm w}_f^T[\theta({\rm x}_i),\phi({\rm x}_j)])
$$
其中 $[\cdot,\cdot]$ 表示级联，${\rm w}\_f$ 是权重向量用于将级联后的向量映射为标量。这种情况下设置标准化因子 $C({\rm x})=N$ 。



以上多种变形表示了 non-local operation 的<u>灵活性</u>。



### Non-local Block

<!--公式有些类似 残差操作-->

将 non-local operation 包装到一个 non-local block 可以结合到多种现有网络结构。定义 non-local  block 为
$$
{\rm z}_i = W_z {\rm y}_i +{\rm x}_i
$$
其中 ${\rm y}\_i$ 为 non-local operation，$+{\rm x}\_i$ 表示 residual connection [^21] 。residual connection 可以使得在任意预训练模型中插入 non-local block 而不会破坏其原始行为，插入时 $W\_z$ 初始化为 0 。

non-local block 结构如下图

{% asset_img non-local-block.png %}

当使用 high-level 、sub-sampled 的 feature maps 时，non-local block 中的二元计算是 lightweight 的。通过矩阵乘法现实的二元计算与标准网络中的典型卷积层相当。



**Implementation of Non-local Blocks**

根据 bottleneck design [^21] 设置 $W\_{g}$ 、$W\_{\theta}$ 和 $W\_{\phi}$ 的 channels 数量为 ${\rm x}$ channels 数量的一半，这能减少 block 一半的计算量。采用下采样方案，将 non-local operation 修改为 ${\rm y}\_i=\frac{1}{C({\rm\hat x})}\sum\_{\forall j} f({\rm x}\_i, {\rm \hat x}\_j)g({\rm \hat x}\_j)$ ，其中 ${\rm \hat x}$ 是 ${\rm x}$ 的下采样结果 ( via. pooling ) 。通过在空间域使用<u>下采样</u>能将 block 计算量减少至 1/4 ，并且这不会改变 non-local 的表现。下采样具体实现是通过在 $\phi$ 和 $g$ 后面添加 max pooling layer 。



## Video Classification Models

### 2D ConvNet Baseline (C2D)

Baseline 采用 50 层的 ResNet 为骨架。输入视频片段为 224*224 是视频帧 32 帧。所有的卷积核为 2D 核 ($1\times k\times k$)， 用于逐帧处理视频。模型参数使用在 ImageNet 上预训练的 ResNet 进行初始化。这与 ResNet-101 的构建方式相似。



### Inflated 3D ConvNet (I3D)

参考论文 [^13][^7] 中的扩充方法，将 2D 卷积核 ($k\times k$) 变化为 3D 卷积核 ($t\times k\times k$) ，其中 3D 核中的有 $t$ 个 planes ，每个 plane 参数取值为原 2D 核参数的 $1/t$ 。

本文研究了两种扩充 3D 核的方式。第一种是将一个残差块中 $3\times 3$ 的核扩充为 $3\times 3\times 3$ 的核，记为 ${\rm I3D}\_{3\times 3\times 3}$。第二种是将一个残差块中的第一个 $1\times 1$ 核扩充为 $3\times 1\times 1$ 的核，记为 $${\rm I3D}\_{3\times 1\times 1}$$。因为 3D 核将使得计算密集型的，所以本文只在每两个残差块上进行一次扩充，过多的扩充会让收益消失。本文还将 ${\rm conv}\_1$ 扩充为 $5\times 7 \times 7$ 。



### Non-local network

通过将 non-local block 插入 C2D 或 C3D 网络中将其转化为 non-local 网络。本文研究了不同的插入 non-local blocks 数量。



**Training**

作者在 ImageNet [^39] 上预训练模型，然后进行 fine-tune。在 fine-tune 的时候默认使用 32 帧视频片段作为输入 。输入片段由 64 帧的原始视频中随机抽取 32 帧，再进行裁剪获得。裁剪过程参考 [^46] ，首先将视频的短边随机缩放到 $[256,320]$ 像素大小，再裁剪为 $224\times 224$ 大小。

作者在 8 块 GPU 上训练，每个 GPU 以 8 个片段为一个 mini-batch ，故整个模型以 48 片段为 mini-batch。迭代次数 400k ，初始化学习率 0.01 ，每 150k 迭代后减少 10 倍，momentum 大小  0.9 ，权重衰减为 0.0001。在全局 pooling 层后使用了 0.5 的 dropout。在 fine-tune 时使用 BN，并且实验显示在作者的应用中使用 BN 能够减少过拟合。

使用 [^20] 的方法初始化 non-local block 的权重参数，并且在 non-local block 中 $W\_z$ 和 ${\rm y}\_i$ 相乘的输出后使用 BN 对数据进行标准化。<u>BN 的输出参数为 0 ，以确保在初始状态是 non-local block 是一种恒等映射，即 ${\rm z}\_i= W\_z {\rm y}\_i+{\rm x}\_i = {\rm x}\_i$ ，所以可以直接插入其他预训练后的网络中。</u>



**Inference**

将视频帧大小缩放到 256 进行全卷积推理。并且对视频随机抽取 10 个片段各自计算出 softmax 分数，使用这 10 个分数的均值作为该视频的最后预测值。



## Experiment on Video Classification

Kinetics数据集，246k 训练视频，20k 验证视频，涉及 400 种人类运动。在该数据集上进行视频分类，训练集用于训练，验证集用于测试。


{% asset_img p1.png %}


首先和 ResNet-50 C2D baseline 对比， 在<u>整个训练过程</u>中，以 top-5 为指标，non-local network-5 的测试误差和验证误差均比 ResNet-50 C2D 要低。（non-local network 能够学习到对于有效的时间和空间上的相对关系）





{% asset_img p2.png %}


表2 列出了各类测试数据。

**Instantiations**

表2a 显示，即使使用一个 non-local block 也能提高 top-5 近 1% 的精度，且 non-local block 中关系函数 $f$ 使用不同的选择，提升的效果是近似的，<u>可见注意力机制只是 non-local operation 的一种形式，而且 non-local operation 是有效的</u>。



**stage**

表2b 显示将单个 non-local block 插入到不同阶段的残差块后的结果，插在不同位置的效果近似，在 ${\rm res}\_5$ 上的效果相对较低。作者解释是因为 ${\rm res}\_5$ 的输出大小 $7\times 7$ 过小，不足以提供精确的空间信息。



**deeper**

表2c 显示通过插入更多的 non-local block ，能够得到更高的准确度。作者据此得出结论，多个 non-local block 可以实现长范围、多跳的依赖连接，信息可以在长距离的时空上进行前向和后向的传递。（局部模型是难以实现该效果的）

且表2c 中显示，50 层的 ResNet 在加入了 non-local 5-block 后的效果优于 101 层的 ResNet baseline，且前者的运算量和参数数量均比后者要少，这验证了 non-local block 是有效性。



**non-local in spacetime**

在视频中，相关的对象可以在大空间距离和长时间间隔中出现，即数据存在时间、空间和时空上的依赖关系。

<u>通过值定 non-local operation 公式中相对位置 $j$ 的范围，可以实现空间、时间、时空三类依赖关系的计算。</u>
<u>比如指定相对位置 $j$ 为与目标位置 $i$ 同帧上的所有位置，可以实现仅捕获空间的依赖。</u>

表2d 显示了 non-local operation 能够有效捕获时间和空间上依赖关系，使得分类准确度提高。



**Non-local netvs. 3D ConvNet**

表2e 显示 non-local operation 和 3D 卷积扩充都是有效的 2D 卷积网络的扩展方式。而 3D 扩充的方式将增加网络的计算量，而 non-local 扩充的方式对模型计算量的增加比 3D 扩充的要少，对准确率的提升更大。



**Non-local  3D  ConvNet**

表2f 显示 non-local 和 3D 扩充结合，可以进一步提高精确度。作者的解释是这两种方式的改善是两个不同的方面。3D 卷积捕获了局部依赖，non-local 捕获的是大范围依赖。



**Longer  sequences**

作者还使用长视频实验验证了模型的范化性能。将原来的训练输入 32 帧每片段改为连续的 128 帧，且没有子采样。由于 GPU 内存限制，将 mini-batch 大小从 8 改为 2，学习率改为 0.0025。

表2g 显示，相对与之前的 32 帧短时长视频，多帧视频训练出的模型可以得到更好的效果，且 NL I3D 模型的效果依然比 I3D 模型要好。验证了该模型在长序列上依然有效。



**Comparisons with state-of-the-art results**

作者对比了该模型与现有最优模型的效果，表示该模型优于其他模型，如表3 。


{% asset_img p3.png %}




作者还在 Charades 数据集上进行了视频分类实验，在 COCO 上进行了对象检测/分割和人类姿势估计的实验，验证了该模型的有效性。





## Personal Opinion

non-local mean $(2)$ 和 non-local operation $(1)$ 计算含义很相似，都是在输入信息上，对某一位置信息计算所有位置信息对它的加权和，用来表示该位置对应的全局关系。  

$$
\begin{aligned}
& NL({\rm I}_i) = \sum_{\forall j} W({\rm I}_i,{\rm I}_j) \cdot {\rm I}_j & \quad(1)  \\
\quad\\
& {\rm y}_i=\frac{1}{C({\rm x})}\sum_{\forall j}f({\rm x}_i,{\rm x}_j)g({\rm x}_j) & \quad(2)
\end{aligned}
$$

而 non-local block $(3)$ 与 ResNet $(4)$ 的残差块的数学表示形式又比较类似。

$$
\begin{aligned}
& H(x)= F(x)+x &(3)\\
\quad\\
&{\rm z}_i = W_z {\rm y}_i +{\rm x}_i &(4)
\end{aligned}
$$

ResNet 中 $F(x)$ 为残差，也是优化的目标。残差块的表示中将残差项 $F(x)$ 和 输入 $x$ 相加，我的理解为这是为了学习表达恒等映射 $H(x)$ 。
而 non-local block 将学习到的 long-range 依赖 ${\rm y}\_i$ 按权加到输入信息 ${\rm x}\_i$ 上（作者的实现中，相加前还对 $W\_z{\rm y}\_i$ 进行了 BN 处理），用于输出。这种方式的输出包含了 long-range 分量，而且可以通过对权重初始化为 0，使得该层可以直接插入结合到任意的预训练网络中。



## Annotation

[^4]: [A non-local algorithm for image denoising - IEEE Conference Publication](https://ieeexplore.ieee.org/abstract/document/1467423)
[^7]: [Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://arxiv.org/abs/1705.07750)
[^13]: [Spatiotemporal Residual Networks for Video Action Recognition](http://papers.nips.cc/paper/6432-spatiotemporal-residual-networks-for-video-action-recognition)
[^20]: [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html)
[^21]: [Deep Residual Learning for Image Recognition](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)
[^39]: [ImageNet Large Scale Visual Recognition Challenge](https://link.springer.com/article/10.1007/s11263-015-0816-y)
[^46]: [Very deep convolutional networks for large-scale image recognition](https://arxiv.org/abs/1409.1556)
[^49]: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

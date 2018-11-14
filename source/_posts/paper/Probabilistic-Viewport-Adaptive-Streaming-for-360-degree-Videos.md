---
title: Probabilistic Viewport Adaptive Streaming for 360-degree Videos
id: prob-tsp
category: paper
date: 2018-11-14 19:17:49
tags: ['360video', 'motion prediction', 'adaptive']
mathjax: true
---

本文提出了基于 概率视口自适应 的全景视频传输实现

<!-- more -->

问题: How to streaming different viewport-dependent segment ?

[[4-7]](#Annotation) : 基于块的视口自适应流，灵活但对头部预测要求高，预测错误会导致黑块

[[10]](#Annotation) : 固定预测网络，该网络使用传感器和内容相关特征来解决预测问题，缺点是耗时，且需要 x2651/HEVC 支持

[[12]](#Annotation) : 非对称投影流，需要 x264/AVC 支持，不会产生黑块，但预测错误会导致显著的质量下降



本文采用高通的 TSP，起兼容 MPEG OMAF，含有全角度内容不会产生黑块，且提供高质量视口，同时仅需要 x264/AVC 支持。本文方法利用用户方向的概率分布，通过最大化预期质量来预取片段。

**本文方法有两个要点：头部移动预测（概率视口预测模型）和 最优分段预取**



> 论文：Probabilistic Viewport Adaptive Streaming for 360-degree Videos


---

## The Method

一个概率视口预测模型 和 一个最佳期望质量框架

视频传输：将全景视频划分为 N 个不同的**视口相关表示**，并且使用 M 比特编码，按系统时间间隔划分多段，采用 TSP 进行传输。

预测用户方向：将全景视频映射到立方体上，根据用户历史移动轨迹计算每个面上每个像素的观看概率，取平面上所有像素的概率平均值作为平面的观看概率

决策：根据每个平面的观看概率，计算 TPS 表示 的**期望质量**，按**最大化期望质量**的方式选取 **视口相关表示** 来进行预取。



### Problem Formulation

{% asset_img p1.png %}

使用 $i\in \\{1,..N\\}$ 表示 **视口相关表示** ， $j\in{\\{1,...N\\}}$ 表示其比特率。$r\_{(i,j)}$ 表示分片 $(i,j)$ 的比特率。$k\in\\{1,...K\\}$ 表示立方体的 $K$ 个面。

问题描述为寻找一个分片流的序列 $X=\\{x\_{i,j}\\}$ 来**最大化**该传输流的**观看概率**和**图像质量**，其中 $x\_{i,j} =1$ 表示比特率为 $j$ 的第 $i$ 个视口发片被选中为传输流中的一个发片。

使用 $Q^k\_{i,j}$ 表示第 $k$ 个面的平均质量，$P^k$ 表示在第 $j$ 个比特率级别的第 $i$ 个视口的片段的第 $k$ 个面的观看概率， $R$ 为总比特率。该问题用数学表示为

$$
\begin{aligned}
 & \max_{X} \sum_{k=1}^k P^k\cdot \max\{x_{i,j}\cdot Q^k_{i,j}\}  \quad  \\
 & s.t.  \sum_{i=1}^N\sum_{j=1}^M x_{i,j}\cdot r_{i,j} \le R, \\
 & \qquad \sum_{j=1}^M x_{i,j} = 1,x_{i,j} \in \{0,1\}, \forall	i
\end{aligned}
$$

> 注意按公式中的约束条件，对于一个视口 $i$， 只会有一个比特率为 $j$ 的分片会被选择。

由于该算法会选择了多个视口进行传输，所以即使用户方向预测错误，也能通过播放其他视口来缓解用户体验下降。



### Probabilistic Model of Viewport Prediction

**视口预测：**本文使用欧拉角度 偏航(yaw) $\alpha$ 、俯仰(pitch) $\beta$  和滚动(roll) $\gamma$ 来表示用户发方向，使用线性回归模型进行预测。

记 $t\_0$ 为当前时刻，使用最小二乘法计算用户移动趋势。用 $v\_{\alpha}$、$v\_{\beta}$ 和 $v\_{\gamma}$ 表示三个维度的坡度(slop)，则i经过时间 $\triangle t$ 间隔后，三个维度的值如下
$$
\begin{aligned}
\hat \alpha (t_0 + \triangle t) =v_{\alpha}\cdot \triangle t + \alpha(t_0)\\
\hat \beta (t_0 + \triangle t) =v_{\beta}\cdot \triangle t + \beta(t_0)\\
\hat \gamma (t_0 + \triangle t) =v_{\gamma}\cdot \triangle t + \gamma(t_0)
\end{aligned}
$$
**预测错误：** 将线性回归模型的预测错误情况画出来，可见其服从高斯分布 $e\_{\alpha} \sim N(\mu\_{\alpha,\sigma ^2\_{\alpha}})$ 。

{% asset_img p2.png %}


通过拟合曲线，可以学习得到均值 $\mu\_{\alpha}$ 和标准差 $\sigma\_{\alpha}$ 。然后得到头部运动的**概率分布**
$$
\begin{cases}
P_{\rm yaw}(\alpha) = \frac{1}{\sigma_{\alpha}\sqrt{2\pi}} \exp (-\frac{(\alpha-(\hat \alpha+\mu_{\alpha}))^2}{2\sigma^2_{\alpha}})\\
P_{\rm pitch}(\beta) = \frac{1}{\sigma_{\beta}\sqrt{2\pi}} \exp (-\frac{(\beta-(\hat \beta+\mu_{\beta}))^2}{2\sigma^2_{\beta}})\\
P_{\rm roll}(\gamma) = \frac{1}{\sigma_{\gamma}\sqrt{2\pi}} \exp (-\frac{(\gamma-(\hat \gamma+\mu_{\gamma}))^2}{2\sigma^2_{\gamma}})
\end{cases}
$$
预测错误的标准差会随着时间间隔 $\triangle t$ 的增大而增大，通过拟合数据，可以得到标准差与时间间隔之间的关系如下
$$
\begin{aligned}
\sigma_{\alpha}=\delta_{\alpha}\cdot(\triangle t)^2\\
\sigma_{\beta}=\delta_{\beta}\cdot(\triangle t)^2\\
\sigma_{\gamma}=\delta_{\gamma}\cdot(\triangle t)^2
\end{aligned}
$$

> 标准差与时间间隔成二次函数关系

得到三个维度的概率分布后，可以得到用户方向的概率分布
$$
P_E(\alpha,\beta,\gamma) = P_{\rm yaw}(\alpha)\cdot P_{\rm pitch}(\beta)\cdot P_{\rm roll}(\gamma)
$$

> 因为三个维度相互独立，所以直接相乘

得到用户方向的概率发布后，可以计算立方体每个面的观看概率。使用 $(\varphi,\theta)$ 表示球面上的点，使用 $L\_k(\varphi, \theta)$ 表示第 $k$ 个面内的点的集合。为了方便，**本文以平面 $k$ 上所有点的平均取向概率( the average probability of orientations )作为该平面的观看概率 $P^k$**
$$
P^k=\frac{1}{|L_k(\varphi,\theta)|}\cdot \sum_{(\alpha,\beta,\gamma)\in L_k(\varphi,\theta)} P_E(\alpha,\beta,\gamma)
$$




## System and Experiment

### System Implement

系统实现包含三个方面，多媒体制作、HTTP 服务器 和 客户端

* 多媒体制作

1. 使用 360tools [^19] 采用 **TSP 非对称映射 **将 ERP 格式的视频转为 TSP 格式的视频

2. 编码器根据 x264[^13] 和 MP4Box[^20] 对每个视口视频进行分割和编码

3. 设计`视口相关生成器`支持视口自适应流，其通过在 MPEG-DASH 中添加与`标准表示定义`相对应的`径度`和`维度`属性来实现。

* 客户端

使用开源的 MPEG-DASH dash.js 播放器 和 eleVR Web Player 来实现全景视频播放器，并且实现本文的概率视口自适应方法。

由八个部分组成：MPD 解析器、带宽评估器、缓存控制器、方向预测器、视口自适应、QoE驱动的优化器、渲染器和头部位置采集器。



### Experiment Setup

数据集：AT&T [^9]  的视频序列及其 5 组用户头部移动轨迹，时长 3 分钟，分辨率 2880 x 1440，格式 ERP。对视频序列按 1s 间隔进行分割，每个分片的比特率为 {300kbps, 700kbps, 1500kbps, 2500kbps, 3500kbps}，采用 x264 编码。

对比方法：ERP[^16]，其将全景视频当作普通视频处理；TSP[^23]，其使用线性回归预测视口且请求对应的分片，但为采用概率视口预测。

评价因子：stall（失速）、视口PSNR (V-PSNR) 、视口错误

实验结果：

* 长带宽实验（500kbps 到 3000kbps，每个带宽持续20秒）：

1. 比特率上，所有方法都具有相似的表现，因为它们采用了同样的**基于缓存的比特率自适应方法**。本文方法偶尔会达到更高的比特率，因为其在期望质量最大化下可能会预取两个视口表示。
2. 在 V-PSNR 上，ERP最低，TPS 和本文方法类似，但 TPS 存在少数 V-PSNR 很低的点，因为预测错误对其影响很严重。
3. 在比特率、stall 次数、视口错误上，本文方法平均表现最优。

* 真实网络实验：

1. 三者具有相似的比特率表现，本文方法有少数时间达到更高的比特率。
2. V-PSNR 类似上个实验，
3. 本文方法具有最高的带宽利用率、最少的 stall、最少的视口错误和最高的 V-PSNR，且V-PSNR 相对稳定。

* 多种头部移动轨迹实验：

不同移动轨迹下，本文方法均有比较好的表现，其 V-PSNR 比 TSP 高 7% 以上，且具有最少的视口错误。

## Annotation

[^4]: [View-aware tile-based adaptations in 360 virtual reality video streaming - IEEE Conference Publication](https://ieeexplore.ieee.org/abstract/document/7892357/)
[^5]: [HEVC tile based streaming to head mounted displays - IEEE Conference Publication](https://ieeexplore.ieee.org/abstract/document/7983191/)
[^6]: [Spatio-temporal activity based tiling for panorama streaming](https://dl.acm.org/citation.cfm?id=3083176)
[^7]: [Ultra wide view based panoramic vr streaming](https://dl.acm.org/citation.cfm?id=3097899)
[^9]: [Optimizing 360 video delivery over cellular networks](https://dl.acm.org/citation.cfm?id=2980056)
[^10]: [Fixation Prediction for 360° Video Streaming in Head-Mounted Virtual Reality](https://dl.acm.org/citation.cfm?id=3083180)
[^12]: [A measurement study of oculus 360 degree video streaming](https://dl.acm.org/citation.cfm?id=3083190)
[^13]: [x264, the best H.264/AVC encoder - VideoLAN](http://www.videolan.org/developers/x264.html)
[^16]: [360 度全景直播视频的编码器设置 - YouTube帮助](https://support.google.com/youtube/answer/6396222)
[^19]: [Samsung/360tools](https://github.com/Samsung/360tools)
[^20]: [Dash-Industry-Forum/dash.js](https://github.com/Dash-Industry-Forum/dash.js)
[^23]: [Interactive panoramic video streaming system over restricted bandwidth network](https://dl.acm.org/citation.cfm?id=1874184)

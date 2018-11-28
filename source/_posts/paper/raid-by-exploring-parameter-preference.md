---
title: Region-Aware Image Denoising by Exploring Parameter Preference
id: raid-by-exploring-parameter-preference
date: 2018-09-27 13:49:41
mathjax: true
tags: ['Image denoising', 'denoising parameter', 'machine learning', 'region-aware']
---
一种通过探索参数偏好的区域感知图片降噪（RAID）算法。

<!-- more -->

两个基本发现：

* 使用较小的噪声水平作为参数，有时候 比直接使用噪声水平作为参数 能获得更好的效果
* 复杂纹理图片区域的最优降噪参数通常比简单纹理的最优降噪参数要小

由此提出一种，通过寻找 preference parameter ，混合各区域的最佳降噪结果的图片降噪算法。




> 论文：[Region-Aware Image Denoising by Exploring Parameter Preference](https://ieeexplore.ieee.org/document/8421285/)



---



## RAID

将图片按噪声水平分组，图片划分为 $n\times n$ 块处理。降噪算法采用 PGPD

* 计算各个噪声水平的**最佳改变率**，获得**最优降噪参数**，并且定义**理想偏好**（ideal preference）
* 训练**参数偏好模型**
* 使用参数偏好模型对目标噪声图片进行**偏好预测**
* 按参数偏好范围组合降噪结果



### 最佳改变率计算

将噪声图片按噪声水平分组，为不同噪声水平 $\alpha$ 计算最佳的改变率 $\widetilde{r}$ 和最佳降噪参数 $\widetilde{\alpha} = \widetilde{r} * \alpha$

最佳的改变率 $\widetilde{r}$ 选择 $R = \\{0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95\\}$

* 使用 $\alpha$ 和 $\alpha' \in {\alpha \times {R}}$ 作为降噪参数，采用 PGPD 对一张图片进行降噪，得到 8 张降噪结果 $I\_{R}^{1}$ \*1 和 $I\_{R}^{2}$ \*7
* 将 8 个降噪结果按相同的方式划分为 $n\times n$ 个无重叠区域
* 对于每个区域 $j$ ，计算 $\alpha$ 和 $\alpha'$（7种）降噪结果之间的偏好 $P\_{j}$ （7组，每组对应N张噪声图片）
$$
\left\{
\begin{aligned}
S_{j}^{1}=\sum_{p\in I_{j}}| I_{j}(p) - I_{Rj}^{1}(p)|\\
S_{j}^{2}=\sum_{p\in I_{j}}| I_{j}(p) - I_{Rj}^{2}(p)|
\end{aligned}
\right.
$$

$$
P_{j} = \begin{cases}
0,& if \quad S_{j}^{1} \le S_{j}^{2}\\
1,& if \quad S_{j}^{1} \ge S_{j}^{2}
\end{cases}
$$

其中 $I\_{j}(p)$ 为clean image 中第 $j$ 区域的像素，$I\_{Rj}^{1}(p)$  为使用 $\alpha$ 作降噪参数所得降噪结果的第 $j$ 区域的像素，$I\_{Rj}^{2}(p)$  为使用 $\alpha’$ 作降噪参数所得降噪结果的第 $j$ 区域的像素。$S\_{j}^{1}$ 和 $S\_{j}^{2}$  表示像素差的绝对和。（$P\_{j}=0$ 偏好 $I\_{R}^{1}$，$P\_{j}=1$ 则偏好 $I\_{R}^{2}$）$P\_{j}$ 为区域的**理想偏好**

* 根据偏好 $P\_{j}$ 组合 $\alpha'$ 对应的**组合降噪结果** $I\_{j}^{c}$ （7组，每组对应N张噪声图片的混合降噪结果），即为**理想偏好图**
$$
I_{j}^{c} = \begin{cases}
I_{Rj}^{1}, &if \quad P_{j}=0\\
I_{Rj}^{2}, &if \quad P_{j}=1
\end{cases}
$$
* 计算7张 $I\_{j}^{c}$ 的平均峰值信噪比 $PSNR\_{avg}$，选平均信噪比最大的 $\alpha'$ 为**最佳降噪参数**  $\widetilde{\alpha}$，其对应改变率为**最佳改变率** $\widetilde{r}$



伪代码：

```python 
noiseImages = input()
iGroups = sortByNoiseLeve(noiseImages) 

R = [0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95] # len(R) = 7, 改变率集合

for g in iGroups: # 对每个噪声水平计算其最优改变率
  a = noiseLevel(g)
  a_ = [ a * r for r in R ]
  
  PSNRavgDic = {} # ai : PSNRavg
  iDenoisedComs = [] # 每个 a_ 会生成一个混合降噪结果
  for ai in a_:
  	iDenoisedComs = []
  	for img in g:
      iDenoised = PGPD(img, a)
      iDenoisedRegions = segment(iDenoised, n, n) # 分割为 n*n 个区域

      _iDenoised = PGPD(img, ai)
      _iDenoisedRegions = segment(iDenoised_, n, n) # 分割为 n*n 个区域
    
      for j in [region for r in n*n ]: # 选择每个区域用于组合
        P[j] = formulate_3(iDenoisedRegions[j], _iDenoisedRegions[j])
        I[j] = formulate_4(P[j])
      iDenoisedCom = combineRegion(I, n, n, iDenoisedRegions, _iDenoisedRegions) # 混合区域组成降噪图片
      iDenoisedComs.append(iDenoisedCom)
  
    PSNRavgDic[ai] = avg(PSNR(iDenoisedComs)) # 记录 ai 对应的平均 PSNR
    
  # 选去 PSNRavg 最大的 ai 作为最优改变率对应的 a
  PSNRavgDic = dict([val, key] for key, val in dic.items()) # 对调键值对 PSNRavg : ai
  bestR = PSNRavgDic[max(PSNRavgDic.keys())] 
```



### 训练偏好参数模型

对与每张噪声图片 $I^{n}$，选择其两张降噪结果 $I\_{R}^{1}$ 和 $I\_{R}^{2}$（来自 $\widetilde\alpha$） ，两张差异图片 $D=I^{n}-I\_{R}^{1}$ 和 $\widetilde{D}=I^{n}-I\_{R}^{2}$ 

将上述 4 张图划分为 $n\times n$ 个区域，对每个区域提出特征。

特征向量（含 3 部分）：

1. $I\_{R}^{1}$ 和 $I\_{R}^{2}$ 的区域的所有像素 $F\_{a}=\\{F\_{1},F\_{2},..,F\_{n\*n}\\}$ 和 $F\_{b}=\\{F\_{n\*n+1},F\_{n\*n+2},..,F\_{n\*n\*2}\\}$
2. 两张差异图片的区域的所有像素 $F\_{c}=\\{F\_{n\*n\*2+1},F\_{n\*n\*2+1},..,F\_{n\*n\*3}\\}$ , $F\_{d}=\\{F\_{n\*n\*3+1},F\_{n\*n\*3+1},..,F\_{n\*n\*4}\\}$
3. 由于纹理简单区域的像素差异比纹理复杂的一般要小，抽取 $F\_{a}$, $F\_{b}$, $F\_{c}$, $F\_{d}$ 之间的差异作为作为第三个特征，$F\_{e}=\\{F\_{n\*n\*4+1},F\_{n\*n\*4+2},F\_{n\*n\*4+3},F\_{n\*n\*4+4}\\}$

故 $FV = \\{F\_{a}, F\_{b}, F\_{c}, F\_{d}, F\_{e}\\}$

标签 label ：对每个区域计算理想偏好 $P\_{j}$ 作为区域的 label

模型 M ：根据特征向量和标签，使用随机森林分类（RFC ）算法，对每个噪声水平进行训练模型



### 偏好预测

计算噪声图片 $I^{n}$ 的噪声水平，根据噪声水平计算 $I\_{R}^{1}$ 和 $I\_{R}^{2}$ ，并且选择噪声水平对应的参数偏好模型 M。

将图片分化为 $n\times n$ 的区域，计算各个区域的特征向量，使用多个决策树计算参数偏好值，取对应多个决策树数量最多的值为最终的**区域预测参数偏好** $P\_{j}$



### 组合降噪结果

由于直接使用区域预测参数偏好进行区域拼接，生成的结果将具有明显的拼接痕迹，故不用区域预测参数偏好值，而采用**区域预测参数偏好范围** $P$ 进行区域拼接
$$
P = f(\frac{A}{A+B})
$$
$$
f(x)=\begin{cases}
0 & x<1-\lambda\\
\frac{x+\lambda-1}{2\lambda -1} & 1-\lambda \le x \le \lambda \\
1 &x>\lambda
\end{cases}
$$
$A$ 和 $B$ 代表决策结果为 $I\_{R}^{1}$ 与 $I\_{R}^{2}$ 各自的决策树数量。则 $x$ 即代表  $I\_{R}^{2}$  的决策树在所有决策树中的比例。$\lambda$ 为阈值（本文取值0.75）

使用 $P$ 进行区域选择然后拼接所有区域，得到最后的降噪结果





## 实验

**硬件设备：** a 2.40GHz Intel Dual 6 Cores CPU and 16 GB memory.

**降噪算法：** PGPD, BM3D, NLM

**数据集：** BSD500， 400训练，68测试。和另一个具有14张常见图片的数据集作为测试数据集

**噪声：** AGWN，噪声水平 σ×255 ∈ {10, 20, ..., 100}

**评价指标：**降噪结果的 PSNR 和 SSIM, visual effect

**RAID 参数：** n=7, N=400, λ=0.75, bestChangeRate（对于各个噪声水平和各个IDA是待实验计算的）

**实验结果：**

* RAID-PGPD 的 PSNR 和 SSIM 均比 PGPD 高。BM3D, NLM, RAID-BM3D, 和 RAID-NLM 也具有相同结果。 
* 算法耗时：两倍的原 IDA 降噪时间+特征提取时间+特征预测时间+结果混合时间。RAID-BM3D, RAID-PGPD, and RAID-NLM 的平均额外时间分别为 0.408, 0.385, and 0.367 秒。

在不同噪声水平上，256 × 256大小的图片上的数据统计：

| 算法类型 | 原算法 平均执行时间/s | 结合 RAID 平均执行时间为/s | 结合 RAID 平均模型训练时间/h |
| -------- | --------------------- | -------------------------- | ---------------------------- |
| BM3D     | 1.328                 | 3.063                      | 1.358                        |
| PGPD     | 19.419                | 39.224                     | 1.171                        |
| NLM      | 33.019                | 66.405                     | 1.354                        |

~~（可见时间成本很大）~~
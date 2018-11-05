---
title: Color correction for stereoscopic image based on matching and optimization
id: mo_color_correction
date: 2018-10-26 00:22:10
tags: ['color correction', 'stereoscopic image', 'unresolved issues']
mathjax: true
---
结合全局和局部颜色信息，矫正S3D图的颜色差异

使用密集的立体匹配（dense stereo matching）和全局颜色矫正算法来初始化颜色值，并且提高局部颜色的平滑性和全局颜色的一致性。


<!-- more -->


> 论文：[Color correction for stereoscopic image based on matching and optimization](https://ieeexplore.ieee.org/abstract/document/8251900)



---


基于颜色映射函数，颜色矫正可以分为全局和局部两种算法

1. 全局颜色矫正算法对目标图像上所有像素采用相同的颜色映射函数，不适用于颜色丰富的图像	
2. 局部颜色矫正算法对目标图像上不同区域使用不同的颜色映射函数。（对于参考图和目标图之间的匹配很敏感）

本文颜色矫正算法

1. 采用密集立体匹配和全局颜色矫正算法初始化结果图中的颜色值
2. 使用SSIM测量匹配质量，以消除错误匹配对颜色矫正的负面影响
3. 使用高质量的局部匹配区域初始化结果图中的颜色值，低质量的匹配区域使用全局算法初始化
4. 改善局部颜色的平滑性和全局颜色的一致性



## The Method

阶段一：

使用 SIFT flow [^19] （一种密集立体图匹配算法）计算参考图和目标图之间的匹配图

使用 SSIM [^6] 算法计算匹配图和目标图之间的相似度，作为置信图

使用 ACG-CDT [^4] 算法计算全局颜色矫正结果

结合匹配图、置信图和 ACG-CDT 结果图得到初始结果图

阶段二：

使用 SLIC [^20] 分割初始结果图

优化局部颜色平滑性和全局颜色一致性

### 颜色矫正初始化

**Matching image**：使用 SIFT flow 生成，匹配图拥有和目标图相似的结构，而匹配图的像素则来自参考图。匹配图中和参考图之间的非匹配区域，无参考像素，使用纯黑表示。

**Confidence map**：SSIM 算法根据三个方面计算匹配图和目标图的相似度，亮度、对比度、和结构。

**Global resulting image**：根据置信图，使用 ACG-CDT 算法的结果补充或替换匹配图中**无匹配区域**和**误匹配区域**。ACG-CDT 是基于累积概率密度函数的迭代矫正算法，相比全局颜色矫正算法的表现更好且稳定。

**Color value initialization**：
$$
I_{ini}(i,j) = \begin{cases}
I_g(i,j) & W(i,j) < \alpha\\
I_m(i,j) & W(i,j) \geq \alpha
\end{cases}
$$
$I\_g$ 是 ACG-CDT 结果，$I\_m$ 是匹配图，$I\_{ini}$ 为初始结果图，$W$ 是置信图，$\alpha$ 是阈值，取0.6 。



### 颜色矫正优化

使用 SLIC 算法将初始矫正结果图分割为多个区域，每个区域具有相似的颜色、亮度和纹理特征。本优化过程对数据、局部颜色平滑度、和全局颜色一致性这三个方面进行优化。

**数据项**：用于保存初始颜色值，以一个初始颜色值和优化颜色值之间的权重项来表示。
$$
\begin{aligned}
E_1 = \sum_i||I_{R_i} - I'_{R_i}||^2 D_i\\
D_i = \begin{cases}
p_1 & M_i < \beta \\
p_2 & \beta \leq M_i < \gamma \\
p_3 & M_i \geq \gamma
\end{cases}\end{aligned}
$$
其中 $I\_{R\_i}$ 和 $I'\_{R\_i}$ 为区域 $R\_i$ 的初始结和优化颜色值。由于匹配图中的像素通常比 ACG-CDT 结果中的像素更具有一致性，所以根据区域的像素来源，使用 $D\_i$ 给与不同区域不同的权重。$M\_i$ 是区域中来自匹配图的像素于总像素之间的比值，$\beta$ 和 $\gamma$ 是阈值，本文取值 0.5 和 1 ，$p\_1$ 、$p\_2$ 和 $p\_3$ 分别取 1、5、10 。

<span style="color: #f00">**problem：**<span>

1. 像素来自匹配图的多，$M\_i$ 变大，$D\_i$ 变大，$E\_1$ 变大。而其描述是取自匹配图像素多更具有一致性，与公式导致的优化方向矛盾。（优化方向是 E 减小）

**局部颜色平滑度**：测量邻居区域之间的颜色相似度，并且使用颜色相似度作为局部颜色平滑度的权重，
$$
\begin{aligned}
& E_2 = \sum_i\sum_{j\in N^l_i}||I'_{R_i} - I'_{R_j}||^2 C_{ij}L_i\\
& \log C_{ij} = - \frac{||d_c(i,j)||^2_2}{\bar{d_c}\cdot \sigma_c^2} \\
& L_i = \begin{cases}
q_1 & M_i < \beta\\
q_2 & \beta \leq M_i < \gamma\\
q_3 & \gamma \leq M_i
\end{cases}
\end{aligned}
$$
$N\_j^l$ 是区域 $R\_i$ 的邻居区域集合，$C\_{ij}$ 是两个区域的颜色相似度，$||d\_c(i,j)||^2\_2$ 是两个区域的平均颜色之间的 $L\_2$ 标准距离平方，$\bar {d\_c}$  是任意两个区域的颜色距离的平方均值，$\sigma\_c^2$ 取 0.2 。$L\_i$ 类似 $D\_i$ ，当区域中的像素来自 ACG-CDT 结果图时，使用大权重以强化区域的一致性。$q\_1$、$q\_2$ 、$q\_3$ 分别取值 25、15、6 。

<span style="color: #f00">**problem：**<span>

1. 矛盾点：颜色越相似，颜色距离小，$C\_{ij}$ 越大，$E\_2$ 越大。而优化是往 $E\_2$ 减小的方向优化的。
2. 取值 ACG-CDT 结果图像素多时，$M\_i$ 应该减少，$L\_i$ 变大，与其描述相同。可为什么优化方向（方向是减小$E\_2$）是减少取自 ACG-CDT 的像素，其理由是什么。

**全局颜色一致性**：
$$
E_3 = \sum_i\sum_{j\in N_i^g} ||I'_{R_i} - I'_{R_j}||^2 C_{ij} L_i
$$
其中 $N\_i^g$ 是于区域 $R\_i$ 具有最相似颜色的区域的集合，实验中，选择前15个颜色最相似的区域。

<span style="color: #f00">**problem：**<span>

1. 在公式中，区域越相似，$C\_{ij}$ 越大。而优化方向是减少 $E\_3$，所以减少区域的相似性的合理性是什么？$L\_i$ 与匹配图有关，可其在该公式中的意义是什么？

**二次能量最小化**：

优化问题可以表达为二次能量最小化问题
$$
E = \lambda_1 E_1 + \lambda_2 E_2 + \lambda_3 E_3
$$
其中 $\lambda\_1$、$\lambda\_2$ 和 $\lambda\_3$ 取值 1，6，0.5 



## Experiment 

**数据集**：Middlebury 立体数据集 [^21] ，选取100对 S3D 图像。使用 Photoshop CS6 改变每对图像中的一个图的颜色，以制造颜色差异。包括局部和全局颜色失真的调整，涉及亮度、饱和度、曝光度和对比度，最大调整30%

**对比算法**：CG、LCC-SIFT、GPCT、CT-MF 和ACG-CDT ，其中CGPT 和 ACG-CDT 属于全局颜色矫正算法，其余属于局部颜色矫正算法。

**评价因子**：主观观察和 DSCSI[^21]

**实验结果**：LCC-SIFT 对 SIFT 匹配结果很敏感，当匹配不当时会矫正错误；ACG-CDT 属于全局颜色矫正，对局部颜色的一致性不能矫正。而本文方法视觉效果良好，且 DSCSI 数值高于其他方法。



## Conclusion

本文的 S3D 图像颜色矫正算法结合了**密集立体匹配**和**全局颜色矫正算法**，以获得初始结果图，并且对局部颜色平滑度和全局颜色一致性进行了优化，优化包含了三部分。

本文方法克服了两个缺点

1. 全局算法不能有效处理局部颜色差异的问题（解决方法：根据置信图，会选取匹配图中的像素，不会选取到全局算法结果中的像素而产生局部差异问题，全局算法结果只用于填补误匹配和无匹配的像素）
2. 当区域中只有少量的特征点得到匹配时产生的误匹配问题（解决方法：误匹配时，会根据置信图判断，选取全局算法结果中的像素进行填补）



<br>

[^4]: [Automated colour grading using colour distribution transfer](https://www.sciencedirect.com/science/article/pii/S1077314206002189)


[^6]: [Image quality assessment: from error visibility to structural similarity](https://ieeexplore.ieee.org/abstract/document/1284395)


[^19]: [SIFT Flow: Dense Correspondence across Scenes and Its Applications](https://ieeexplore.ieee.org/abstract/document/5551153)


[^20]: [https://infoscience.epfl.ch/record/149300](https://infoscience.epfl.ch/record/149300)


[^21]: [vision.middlebury.edu/stereo](http://vision.middlebury.edu/stereo)
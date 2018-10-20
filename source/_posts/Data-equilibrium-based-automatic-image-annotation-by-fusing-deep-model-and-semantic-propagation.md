---
title: >-
  Data equilibrium based automatic image annotation by fusing deep model and
  semantic propagation
id: image_annotation_by_deep_model_and_semantic_propagation
date: 2018-10-20 20:35:03
tags: ['image annotation', 'SAE', 'semantic propagation']
mathjax: true
---


提出了一种标注模型和一种标注过程

1. 标注模型用于提高低频率标签的标注效果，提出了一种BSAE用于强化低频标签的训练，RBSAE算法用于强化子BSAE模型的分组训练，以提高稳定性。这种策略保证了模型处理不均衡数据的能量。

2. 对于标注过程，提出ADA框架。对未知图像构造一个局部平衡的数据集，区分图像的高低频属性以决定对应的标注过程。LDE-SP算法用于标注低频图像，RBSAE 算法用于标注高频图像。该策略改善了整体图像标注效果，保证了标注过程对不均衡数据的处理能力。

对SAE提出了两种优化方式，线性优化和非线性优化。

<!--more-->

> 论文: [Data equilibrium based automatic image annotation by fusing deep model and semantic propagation](https://www.sciencedirect.com/science/article/pii/S0031320317302091)



## intro

早期的图像标注研究主要使用概率和统计方法，

'word co-occurrence' 模型建立每个图像单元类型和对应关键字之间的概率分布

translation 模型聚类图像区域的划分，构造视觉关键字词汇。相关模型假设对应于相同关键字的视觉特征是一致的。图像可以通过建立视觉概率分布和语义关键字的链接来进行标注。问题在于难以在实际环境中建立准确的频率模型。

近几年的图像标注研究主要聚焦于基于机器学习的标注和基于图的的标注。矩阵matrix和张量tensor的分解理论也用于图像标注问题。

现存方法的两个问题：

1. 复杂度高且效率低。基于KNN的方法不仅需要计算训练集中每对图像的相似度，还需要计算训练即和测试集中图片的相似度，基于图模型的方法则需要计算复杂的图结构，新增节点时需要许多遍历操作。
2. 在不平衡数据上，低频率图像的标注效果差

对于问题1，本文提出使用SAE训练多标签分类，当分类器训练完成后能够快速对未知图像进行标注。

对于问题2，本文提出标注模型（annotation model）和标注方法（annotation process）来解决。

DBN 和 SAE 均使用逐层的预训练和微调。其中DBN使用概率和统计方法进行预训练，且优化的限制条件很多；SAE网络关注特征之间的近似表达式，能够通过调整模型简单的用于特定环境，来将复杂输入表示为理想的输出。

SAE 优化理论：

1. NL-SAE，通过构造高阶损失函数，使用微分计算最优值。（该方法可以通过控制训练过程的迭代次数来防止过拟合；然而卷积速度慢，且不适合中小型规模数据集）
2. L-SAE，构造一般的线性系统，通过逼近线性方程组的解来获得最优值（可以用最大估计法直接获得更精确的解，无需迭代，可快速训练中小型数据集；然而由于计算过程需要基于样本数量构造矩阵，不适合大型数据集）



## Tradition auto-encoder

将图像标注任务视为多标签分类任务。使用图像特征作为模型输入，用图像标签作为监督信息。首先使用AE预逐层训练网络权重，然后使用权重初始化深度神经网络，最后训练深度模型以最小化训练误差。

### problem definition

记 $X=\\{x\_1,x\_2..x\_N\\}, N\in R^d$ 为N个训练图像， $Y=\\{y\_1,y\_2..y\_M\\}$ 为M种可能的关键词。将图像标注问题表达为 $P=\\{(x\_1,y\_1),(x\_2,y\_2)...(x\_N,y\_N)\\}, Y\_i \in Y$。 

本文记 $Y\_i\in \\{0,1\\}^M$ 为 M 维向量，$Y\_i^j =1$ 表示第 i 张图被关键字 $y\_j$ 标注。为0表示未被该关键字标注。 

### NL-SAE

AE 由编码 $f\_\theta$ 和解码 $g\_{\theta '}$ 两部分组成，前者将输入向量 $x$ 转换为隐含的表示 $h$，后者将 $h$ 重构为输入空间上的向量 $x'$，并且重构过程由损失函数 $L(x,x')$ 进行优化。
$$
\begin{aligned}
f_\theta (x) = \sigma (W \cdot x + b)\\ \sigma(x)=\frac{1}{1+e^{-x}}
\end{aligned}
$$
其中 $\theta =\\{W,b\\}$ 为矩阵权重，且 $W'=W^t$ （正交阵，$W'\cdot W^T=E$ 且矩阵每列长为1）。$b$ 是偏向量，$\sigma$ 为 Sigmoid 激活函数。
$$
g_{\theta'}=\begin{cases}
\sigma(W'\cdot h +b') & x\in[0,1]\\
W'\cdot h + b' & x\in R\end{cases}
$$
其中 $\sigma'=\\{W',b'\\}$

AE 用于学习估计值 使得 $x'=g\_{\theta'}(f\_\theta(x))$ 尽可能于 $x$ 相近。本文定义损失函数 $L(x,x') = (x-x')^2$。通过最小化损失函数学习模型
$$
\theta^\star,{\theta'}^{star} =\mathop{\arg\min}_{\theta,\theta^\star} \frac{1}{N}\sum_{i=1}^L(x_i,g_{\theta'}(f_\theta(x_i))
$$
考虑用带有 L个隐藏层的 NL-SAE 做图像标注。定义 $l\in [1,..,L]$ 为隐藏层的下标索引。 $h^l$ 为第l层的输出向量，（ $l^0 = x$ ），$W^l$ 和 $b^l$ 分别为 l 层的权重和偏差。前向反馈如下
$$
h^{l+1} = \sigma(W^{l+1}h^l+b^{l+1})\quad l\in \{0, ...,L-1\}
$$
使用BP算法 [^BP algorithm] 训练整个模型
$$
\theta^\star = \mathop{\arg\min}_{\theta}\sum_{i=1}^N L(F_\theta(x_i), Y_i)
$$
其中 $F\_\theta(x)=\sigma\_{\theta\_L}(...(\sigma\_{\theta\_1}(x)))$ 是 NL-AE 函数的复合，且 $\theta\_l$ 是模型参数 $\\{W^l,b^l\\},l\in\\{1,..L\\}$，损失函数取 $L(x,y) = (x-y)^2$



### L-SAE

多层极限学习机（multi-layer extreme learning machine）通过逐层叠加线性 AE 来构造多层神经网络。给定 N 个训练样本 $\\{(x\_1,Y\_1),..(X\_N,Y\_N)\\}$ 和隐藏层的神经元个数 $\tilde{N}$ 。EML 可以通过如下公式解学习问题
$$
HB=T
$$
其中
$$
\begin{aligned}
&  H = \begin{bmatrix}
g(w_1\cdot x_1+b_1) & \cdots & g(w_{\tilde{N}}\cdot x_1+b_{\tilde{N}})\\
\vdots & \ddots & \vdots\\
g(w_1\cdot x_N+b_1) & \cdots & g(w_{\tilde{N}}\cdot x_N+b_{\tilde{N}})
\end{bmatrix}_{N\times \tilde{N}},\\
& B = \begin{bmatrix}B_1^T \\ \vdots \\ B_\tilde{N}^T\end{bmatrix}_{\tilde{N}\times M},
\qquad T = \begin{bmatrix}Y_1^T \\ \vdots \\ Y_\tilde{N}^T\end{bmatrix}_{\tilde{N}\times M},
\end{aligned}
$$
其中 $x\_i=[x\_{i1}, \cdots X\_{iD}]^T$ 是第 i 个样本，$w\_i=[w\_{i1},\cdots W\_{iD}]^T$ 是连接第 i 个隐藏神经和输入神经的权重向量， $b\_i$ 是第 i 个隐藏神经的变差。$B\_i = [B\_{i1},\cdots B\_{iM}]$ 是连接第 i 个神经和输出精神的权重，$Y\_i=[Y\_{i1}\cdots Y\_{iM}]$ 第 i 个样本的标签向量，$g$ 是 Sigmoid 激活函数。使用如下公式计算B
$$
B= H^+T
$$
其中 $H^+$ 是 $H$ 的 Moore-Penrose广义逆矩阵 [^Moore-Penrose generalized inverse] 。当 $H^TH$ 是非退化的时候，$H^+=(H^TH)^{-1}H^T$，当 $HH^T$ 是非退化时， $H^+=H^T(HH^T)^{-1}$

ELM 经过修后可以用于 非监督学习：将输入作为输出 $Y=x$，为隐藏节点选取正交随机权重和变差。正交参数可以提高ELM-AE的泛化性。

输出权重B的计算:
$$
\begin{gather*}
 B = (\frac{I}{A}+H^TH)^{-1}H^TX\\
 or \\
B=H^T(\frac{I}{A}+HH^T)^{-1}X\end{gather*}
$$
其中 $I$ 是单位矩阵， $A$ 是用户指定参数，$H=[h\_1,\cdots h\_N]$ 是ELM-AE的隐藏层输出向量， $X=[x\_i,\cdots x\_N]$ 是输入和输出向量。

多层 ELM 的隐藏层权重使用线性 AE 初始化，第 $l-i$ 层的输出和第 $l$ 层的输出有如下关系
$$
H^l = g((B^l)^T H^{l-1})
$$
$B^l$ 是 第 $l$ 层的权重，$g$ 是 Sigmoid 函数。



## Boosted auto-encoder

传统 SAE 整体效果不好的两个理由

1. 数据不均衡。高频标签训练良好，低频标签训练不足，导致低频标签准确率远低于高频标签
2. 简单的 SAE 模型具有很多参数。当参数变化，标注效果会改变，缺少健壮性，难以使用。

对此提出了 BSAE（NL-BSAE 和 L-BSAE）和 RBSAE（NL-RBSAE 和 L-RBSAE）

{% asset_img BAE.png [BAE] %}

### NL-BSAE

$\phi(x)$ 判断样本 $x$ 中低频率标签的数量是否小于阈值 $k$ ，小于则**对样本加入适当的噪声**。

$\varphi(i)$ 表示样本 $x$ 的训练强度，若低频标签数量小于阈值 $k$，则**增加样本的训练次数**。

得到以下式子
$$
\theta^\star,\theta'^\star = \mathop{\arg\min}_{\theta,\theta'}\frac{1}{N}\sum_{i=1}^N\{\frac{1}{\varphi(x_i)}\sum_{j=1}^{\varphi(x_i)}L(\phi(x_i),g_{\theta'}(f_\theta(\phi(x_i))))\}
$$
记 $C=(c\_1,c\_2\cdots c\_M)$ 为训练集 $P$ 中所有关键词出现的次数，$c\_i$ 对应关键词 $y\_i$，使用 $\Pi=\frac{1}{M}\sum\_{j=1}^{m}c\_j$ 表示所有关键词的评价出现次数。得到 $Y\_{C,I} =C * Y\_i$ 表示训练集上每个关键词 $Y\_i^j\quad (j\in \\{1, ..M\\})$ 在第 $i$ 个图像上出现的次数。使用$\Delta\_i=Min(Y\_{C,i}^j)$ 表示在图像 $x\_i$ 上具有最小出现次数的关键词。得到如下
$$
\varphi(x_i) = \begin{cases}
\alpha \cdot \frac{\Pi}{\Delta_i} = \alpha \cdot \frac{\frac{1}{M}\sum_{j=1}^M c_j}{\min\limits_j(Y_{C,i}^j)}, & \Delta_i <= \beta\cdot \Pi\\
1, & Others
\end{cases}
$$
其中 $\alpha$ 和 $\beta$ 是常数，后者用于检测检验是否需要增强训练，前者用于控制需要增强训练的样本的训练强度。
$$
\phi(x_i) = \begin{cases}
\chi \cdot (\frac{1}{d}\sum_{j=1}^d x_i^j) \cdot Ran(\cdot),&\Delta_i<=\beta\cdot \Pi\\
x_i, &Others
\end{cases}
$$
$\chi$ 是用于控制所添加噪声的强度的常数，$d$ 是特征 $x\_i$ 的维度，$x\_i^j$ 是第 $j$ 维度的值，$Ran(\cdot)$  用于生成于 $x\_i$ 相同维度的随机向量，随机向量的每个部分是标准高斯分布或0到1的均匀分布。

当权重训练好后，使用BP算法微调整个模型，而指定样本的训练在微调过程中加重，即
$$
\theta^\star= \mathop{\arg\min}_{\theta}\sum_{i=1}^N\sum_{j=1}^{\varphi(x_i)}L(F_{\theta}(\phi(x_i)),Y_i)
$$
在 NL-BSAE 中，会增强包含低频标签的样本 $x\_i$ 的训练，即隐式扩大了数据集中具有低频标签的图像。能够强化度低频关键词的标注准确率。



### L-BSAE

对于线性BSAE，同样引入 $\phi(x)$ 和 $\varphi(x)$ 的概念，然后有 $\tilde HB = \tilde T$
$$
\begin{aligned}
&  H = \begin{bmatrix}
g(w_1\cdot \phi(x_1)+b_1) & \cdots & g(w_{\tilde{N}}\cdot \phi(x_1)+b_{\tilde{N}})\\
\vdots & \ddots & \vdots\\
g(w_1\cdot \phi(x_N)+b_1) & \cdots & g(w_{\tilde{N}}\cdot \phi(x_N)+b_{\tilde{N}})
\end{bmatrix}_{(\sum\limits_{j=1}^N\phi(x_i))\times \tilde{N}},\\
& B = \begin{bmatrix}B_1^T \\ \vdots \\ B_\tilde{N}^T\end{bmatrix}_{\tilde{N}\times M},
\qquad T = \begin{bmatrix}Y_1^T\\ \vdots \\ Y_{\varphi(x_i)}^T \\ \vdots \\ Y_N^T\\ \vdots \\Y_{\varphi(x_N)}^T\end{bmatrix}_{(\sum\limits_{j=1}^N\varphi(x_i))\times M},
\end{aligned}
$$
其中 $\sum\_{i=1}^n\varphi(x\_i)$ 是所有样本的训练总强度。然后有 
$$
\begin{gather*}
B = (\frac{I}{A}+\tilde H^T \tilde H)^{-1}\tilde H^T \tilde X\\
 or \\
B=\tilde H^T(\frac{I}{A}+\tilde H\tilde H^T)^{-1}\tilde X\end{gather*}
$$
其中 $\tilde H = [h\_1,..h\_{\varphi(x\_1)},..h\_n,..h\_{\varphi(x\_N)}]\_{1\times(\sum\_{i=1}^n\varphi(x\_i))}$ 是 AE 隐藏层的输出，且 $\tilde X = [\phi(x\_i),..\phi(x\_N)]$ 是输入和输出向量。同样有
$$
\tilde H^l = g((B^l)^T\tilde H^{l-1})
$$


{% asset_img RBSAE.png [RBSAE] %}

### NL-RBSAE

BSAE 提高了低频关键词的标注精度，可是过于复杂，需要很多的参数。每个参数都会结果有很大影响，差的设置会使得整个模型效率低下。为此提出了RBSAE

1. 获取训练集图像特征，按**加噪方式**对图像分组，每组分别对所有数据训练 $BSAE\_k^t$ 子 模型
2. 计算 $BSAE\_k^t$ 子 模型的分类错误率，获得具有最小错误率的模型 $BSAE^t$
3. 根据错误率计算每组 $BSAE^t$ 的权重，按权重结合 $BSAE^t$，得到加权 BSAE 模型
4. 使用加权 BSAE 模型标注新输入的图片
5. 生成预测分布 $D$，对 $D$ 进行排序得到结果

根据加噪方式进行分组，每组对应特定的加噪方式。 每组中的子模型 $NL-BSAE\_k^t$ 根据隐藏层神经元的数量不同进行划分。$t$ 代表第 t 种加噪方式，$k$ 是第 k 个 NL-BSAE 模型的隐藏层神经元的数量（**神经元数量为层数 x 每层的神经元个数**）。

需要设置训练数据集的权重来计算每组中 $NL-BSAE\_k^t$ 分类错误率和每组中最优 $NL-BSAE^t$ 的权重。初始权重如下
$$
W_1 = (w_{11},\cdots w_{1i}, \cdots w_{1N}), \quad w_{1i} =\frac{1}{N}, i\in[1,..N]
$$
模型 $NL-BSAE\_k^t$ 的分类错误率为
$$
\begin{gather*}e_k^t =\sum_{i=1}^N w_{ti}\cdot Sgn(NL-BSAE_k^t(x_i) \neq Y_i),\\
Sgn(x) = \begin{cases}1, & x=ture\\ 0, & x=false\end{cases}\end{gather*}
$$
假设图像 $x\_i$ 实际的标签集 $Y\_i$ 包含 $c$ 个关键词，使用 $NL-BSAE\_k^t$ 模型预测标签并且选择同样数量的标签组成预测标签集 $Y\_i^*$。若 $Y\_i\Neq Y\_i^*$，则 $NL-BSAE\_k^t(x\_I) \Neq Y\_i$ 为 $ture$，否则为 $false$

 在每组中选择具有最小分类错误率 $e^t$ 的子模型为 $NL-BSAE^t$，其权重为
$$
\alpha^t = \lambda \cdot \log \frac{1-e^t}{e^t}
$$
其中 $\lambda$ 提出取常数 0.5。

当第 t 组模型训练好后，更新训练集权重去获取下一组模型的权重
$$
\begin{gather*}W_{t+1} = \{w_{t+1, 1}, \cdots w_{t+1, i}, \cdots w_{t+1,N} \}\\
w_{t+1,i} = \frac{w_{ti}\cdot e ^(-\alpha \cdot Y_i\cdot NL-BSAE^t(x_i))}
{\sum\limits_{j=1}^{N}w_{ti}\cdot e^(-\alpha\cdot Y_i\cdot NL-BSAE^t(x_i))},\quad i=1,2..N\end{gather*}
$$
训练完所有组后，计算关键词预测分布
$$
D = \sum_{t=1}^T\alpha^t\cdot NL-BSAE^t(x)
$$

**Problem of Formulation**

* 为什么同组内数据（即每张图）需要权重 $W\_t$ ？
* 为什么组 t 的数据权重与前一组 t-1 的数据权重有关？（为什么一张图在第 t 组的权重与其在第 t-1 组的权重有关）？组间差异是仅是各组数据的加噪方式不同，为什么导致数据权重也不同？
* 权重关系中 $Y\_i\cdot NL-BsAe^t(x\_i)$ 是什么含义，二者为关键词向量，列向量为什么能点乘？
* ~~每组的最优子模型的权重计算公式实际含义是什么？$\alpha^t$ 中 $t$ 为组号，为什么会带入计算？~~ $e^t$ 为模型错误率，错误率小的时候，该计算所得值大，权重变大，合理。
* 每组的数据集不同（加噪方式不同），为什么要将每组的 $NL-BSAE^t$ 取不同权重，求和得最终的加权模型。
* 对数据集进行多种加噪后，再对每种加噪结果进行 LN-BSAE（即组内多个LN-BSAE），复杂度和时间岂不是成倍增加？



### L-RBSAE

1. 获取噪声样本，训练所有组的 L-BSAE 子模型
2. 计算子模型的错误率，得到每组的最优模型 $L-BSAE^t$
3. 使用错误率计算每组 $L-BSAE^t$ 的权重，得到加权的 L-BSAE
4. 使用加权 L-BSAE 对新图像进行标注
5. 生成预测分布 $D$，通过排序得到最后标注结果

$L-BSAE^t\_k$ 中 $t$ 为第 t 中加噪方式，$k$ 是第 k 种 L-BSAE 模型的**初始权值设置**。

同样需要对训练数据设置初始化权重，与 NL-RBSAE 相同为 $w\_{1i} = 1/N$，模型的分类错误率、训练数据集的权重更新方式、预测分布 $D$ 的计算均与 NL-RBSAE 相似，




## Decision based on data balancing 

中低频标签训练不足的本质为改变，为此提出 data equilibrium-based image annotation

1. 为测试图片构造局部均衡的数据集
2. 使用该数据集识别测试图像的属性（属于高频还是低频）
3. 若测试图像为高频图像，使用RBSAE算法进行标注，否则使用局部均衡数据集，按语义传播算法进行关键词预测

### LDE-SP

#### local equilibrium database

构建理想的局部平衡数据集，该数据集满足下列条件

1. 该数据集含有原数据集所有的标签
2. 各标签的频率是相近的
3. 标签所对应图像具有表达标签语义含义的能力

构建数据集方式：

1. 将具有相同 tag 的图像划分为同个语义组。各组元素（即图像）集合之间的交集允许不为空
2. 移除语义组中的弱语义图像
3. 对测试图像 I，每语义组选出前 n 个具有最高相似度图像，构成该数据集

定义弱语义的方式：

定义 $G=\\{(y\_1,X\_1),..(Y\_M,X\_M)\\}$，$y\_i$ 表示第 i 个tag， $X\_i$ 为具有第 $i$ 个标签的图像集合。使用 $G\_i=(Y\_i,X\_i)$ 表示一个语义组。当组 $G\_i$ 的图像数量大于 所有语义组的平均图像数量 $p$ 倍时，该语义组的不同关键字的 co-occurrence 矩阵为
$$
CoMatrix = \begin{bmatrix}
v_{11}&v_{12}&\cdots &v_{1Max}\\
v_{21}&v_{22}&\cdots &v_{2Max}\\
\vdots &\vdots &\vdots &\vdots \\
v_{Max1}&v_{Max2}&\cdots &v_{MaxMax}\\
\end{bmatrix}_{MaxMax}
$$

其中 $v\_{ij}$ 是关键字 $y\_i$、 $y\_j$ 的共同出现次数， $Max$ 是该语义组的关键词种类数。使用如下定义关键字 $y\_i$ 与该语义组的关系
$$
R_{y_i} = \sum_{j=1}^{Max} v_{ij}
$$
然后选出 t 个关系值最小的关键字，并且移除这些关键字所对应的图像，以获得新的语义组 $\hat{G}=\\{(y\_1,\hat X\_1),..(y\_m,\hat X\_M)\\}, \hat X\_i \in X\_i$

#### sematic propagation algorithm 

用条件概率模型 $P(x|y\_i)$ 衡量（给定标签 $y\_i\in Y$ 的）图像 $x$ 的特征分布。然后将图像标注视为后验概率问题（posterior probabilities）
$$
P(y_i|x) = \frac{P(x|y_i)P(y_i)}{P(x)} \propto P(x|y_i)
$$

> 已知语义组 $y\_i$ 中出现 $x$ 图像的概率（ $P(x|y\_i)$ ），求图像 $x$ 属于语义组 $y\_i$ 的概率 （ $P(y\_i|x)$ ）

对于图像 $I$ 的预测最佳标签由如下给出
$$
y^* = \mathop{\arg\max}_i P(y_i|I)
$$

具体步骤：

1. 从语义组 $\hat G\_i$ 中选 n 个与图像 $I$ 最相似的图片，组成新的语义组 $\hat G\_{I,i}$
2. 组合各个语义组 $\hat G\_{I,i}$ 得到 $\hat G\_{I} = \\{\hat G\_{I,1} \cup .. \cup \hat G\_{I,M} \\}$
3. 得到给定标签 $y\_k \in Y$ 下图像 $I$ 出现的后验概率
4. 由后验概率计算 $P(y\_i|I)$，取最小值得到结果
$$
P(I|y_k)  =\sum_{(y_i,\hat X_i) \in \hat G_I} \theta_{I,x_i} \cdot P(y_k|x_i)
$$
$$
\theta_{I,x_i} = \frac{e^{-Dis(I,x_i)} - e^{Dis(I,x_i)}} {e^{-Dis(I,x_i)} + e^{Dis(I,x_i)}} + \mu
$$

其中 $P(y\_k|x\_i)$ 取值为 0 或 1 ，当图像 $x\_i$ 的标签为 $y\_k$ 时取1，否则取0，$\theta\_{i,x\_i}$ 是 图像 $x\_i$ 的权重，$Dis(I,x\_i)$ 是图像 $I$ 和 $x\_i$ 的距离 [^7] ， $\mu$ 为常数取1


{% asset_img ADA.png [ADA] %}


### ADA

1. 根据关键词将训练集分为语义组
2. 根据测试图像构造局部平衡数据集
3. 训练 RBSAE 模型（先训练BSAE模型，再进行 RBSAE 算法）
4. 从语义组和局部均衡数据上分别得到 全局词频率信息 和 局部词频率信息
5. 使用上述两种词频率信息描述待测试图像的属性
6. 取全局和局部的高频率信息的交集，若交集中高频词数量大于阈值，将待测图像标为高频，标记为中低频
7. 若待测图像为高频，采用 RBASE 预测，否则进行语义传播算法预测。

对测试图构建局部均衡数据集 $\hat G\_I$，从中选出 m 张于图像 $I$ 最相似的图像，构成数据集 $\tilde G\_I =\\{(y\_1,\tilde X\_1),(y\_i,\tilde X\_i),..(y\_m,\tilde X\_M)\\}$，其中 $\tilde X\_i$ 可能为空，因为是从所有 $X\_i$ 中取最近似的 m 张图。记 $\tilde X\_i \in \\{0,1\\}^N$ 为 N 维向量，$\tilde N\_i^j$ 为1 表示 $\tilde X\_i$ 包含图像 $x\_j$，否则表示不包含。由此得到预测部分 $D$
$$
D=\begin{cases}
\sum_{t=1}^T \alpha ^t\cdot BSAE^t(x) ,& (\sum_{(y_i,\tilde X_i)\in \tilde G_I} \Phi(y_i)\cdot \Theta(\tilde X_i)) \geq\varepsilon \\
P(y_k|x) ,& Others
\end{cases}
$$
其中 $P(y\_k|x) \propto p(x|y\_k) = \sum\_{(y\_i,\tilde X\_i) \in \tilde G\_i} \theta\_{i,x\_i}\cdot p(y\_k|x\_i)$ , $\varepsilon$ 是常数， $\Phi(y\_i)$ 用于确定关键词在集合 $P$ 中的属性
$$
\Psi(y_i)=\begin{cases}1, & c_i \geq \eta\cdot \frac{1}{M}\sum_{j=1}^M c_j\\
0, & Others
\end{cases}
$$
其中 $\eta$ 是常数， $\Psi(y\_i)=1$ 表示 $y\_i$ 是数据集 $P$ 的高频关键词。

$\Theta(\tilde X\_i)$ 用于分离出 $\tilde G\_I$ 中的高频关键词
$$
\Theta(\tilde X_i) =\begin{cases}1,& \sum_{j=1}^M\tilde X_j^i \geq \tau \cdot (\frac{1}{M}\sum_{i=1}^{M}\sum_{j=1}^{N} \tilde X_i^j)\\
0, & Others \end{cases}
$$

其中 $\tau$ 是常数，$\Theta(\tilde X\_i)=1$ 表示 $y\_i$ 是集合 $\tilde G\_I$ 的高频关键词。

如果 m 给图像中具有高频关键字的图像比例超过合适的数量，图像 $I$ 的结果由 $\sum\_{t=1}^t \alpha ^t\cdot BsAe^t(x)$ 决定，否则由 $P(y\_k|x)$ 决定。

当 RBSAE 采用 NL-RBSAE ，该框架为 NL-ADA 由良好的泛化性，适用于**大规模**数据；当采用 L-RBSAE，框架为 L-ADA，能快速计算结果适用于**中小规模**数据。



## Experiments

**数据集**：Corel5k, Espgame, Iaprtc12

**图像特征**：全局和局部特征（取自论文 [^7] ），其中全局特征有RGB, HSV and LAB color spaces, the Gist features ;对于 SAE、BSAE、RBSAE，使用总共 3312 维的 DenseHue, DenseHueV3H1, DenseSift, Gist, HarrisHue, HarrisHueV3H1, 和 HarrisSift 特征。使用15个特征来构建局部均衡数据集和实现语义传播算法，对于两个特征的距离，使用 $L\_1$ 测量颜色直方图，$L\_2$ 测量 Gist，$\chi^2$ 用于 Sift 和 Hue 的描述。

**评估因子**：取自论文 [^7] 

1. 标注前五个相关关键词
2. 对每个关键词 $y\_j$ 计算精度 $P^j$ 和 recall $R^j$

$$
P^j=\frac{Precision(y_j)}{Prediction(y_j)}\\
R^j=\frac{Precision(y_j)}{Ground(y_j)}
$$
其中 $Precision(y\_j)$ 为预测正确的关键词 $y\_j$ 次数，$Prediction(y\_j)$ 为预测到关键词 $y\_j$ 的所有次数，$Ground(y\_j)$ 是关键词 $y\_j$ 的 ground-truth 次数。
平均精度和所有关键词的 recall rate 分别为 $P$、$R$
$$
\begin{gather*}
P = \frac{1}{M}\sum_{j=1}^M P^j\\
R = \frac{1}{M}\sum_{j=1}^M R^j\end{gather*}
$$

使用 $F\_1$ 测量 $R$ 和 $P$ 的平衡，使用非零 recall $N^+$ 反应关键词数量（有多少关键词预测正确和不正确）
$$
\begin{aligned}
& F_1=\frac{2\cdot P \cdot R}{P+R}\\
& N^+=\sum_{j=1}^M Sgn(R^j)\\
& Sgn(x)=\begin{cases}1, &x>0\\0, & x=0\end{cases}
\end{aligned}
$$



### 实验结果

对 SAE 使用四层网络结构，对比 SAE、NN、BDN、CNN 后，从结果上看 SAE 最有效。CNN 效果不好是因为训练集有限且不均衡，导致特征提出不理想。BDN 于 SAE 效果相似，当 BDN 采用概率和统计方式进行逐层预训练，优化问题的限制很多。SAE 使用近似表达来表示特征，更灵活。

实验参数 $\alpha$ 取 1，$\beta$ 取 0.5 。 对 $\chi$ 选取 1 到 2 的值进行实验，得到其在 1.2 或 1.5 时有最优结果。且 NL-BSAE 效果比 L-BSAE 效果好，因为后者无法控制拟合度，当测试集与训练集差异较大时效果就较差。

使用标签平均频率划分数据集为高频和低频标签，计算 $F\_1$ 和 $N^+$，BSAE 效果比 SAE 效果好。

使用加噪 $Rand(\cdot)$ 测试 RBSAE 的效果，RBSAE 在 $N^+$ 和 $F\_1$ 的相对稳定性上更好。

对比 RBSAE 和 LDE-SP，在高频标签上前者的 $F\_1$ 比后者高，在低频标签上前者的 $F\_1$ 比后者低。在三个数据集上对比 ADA 和近年的模型和经典模型，结果显示了 ADA 的有效性。

时间复杂度上，NL-BSAE 和 NL-RBSAE 的时间与训练样本成正比（大于 NL-SAE ）。L-BSAE 的时间成本高与 L-SAE，且随样本增加，所需时间是非线性增加，在小规模数据上，L-BSAE 比 LN-BSAE 快，而大规模数据上，则比后者慢。在测试时间方面特点，与训练时间特点基本一致。


## Conclusion
BSAE 提高**不均衡数据**的训练效果
RBASE 提高 BSAE 的**稳定性**
BSAE 的两种优化方式，非线性和线性，分别适用于**大规模数据**训练和**小规模数据**训练
ADA 框架使用 结合 RBSAE 和 LDE-SP ，用于**高频和低频图**像标注，提高低频图像标注准确率





## word & reference	

AE: AutoEncoder， [AutoEncoder详解 - lwq1026的博客 - CSDN博客](https://blog.csdn.net/lwq1026/article/details/78581649)

SAE: [Stacked Autoencoder - CoderPai的博客 - CSDN博客](https://blog.csdn.net/CoderPai/article/details/78941411)

BSAE: a balanced and stacked auto-encoder 一种均衡堆叠的自动编码器 

RBSAE: a robust BSAE 一种健壮的均衡堆叠自动编码器

DAE: a stacked denoising auto-encoder 堆叠降噪自动编码器

NL-SAE: non-linear-based stacked auto-encoder 非线性SAE

ADA: attribute discrimination annotation 属性区别标注

LDE-SP: local semantic propagation 局部语义传播

DBN: deep belief networks 

ELM: exterme learning machine [极限机器学习](https://zh.wikipedia.org/wiki/%E6%9E%81%E9%99%90%E5%AD%A6%E4%B9%A0%E6%9C%BA)

LDE-SP: Local data equilibrium and semantic propagation 局部数据均衡和语义传播

ADA: Attribute discrimination annotation 

Gist: [机器视觉-GIST特征描述符使用 - 扬子落木 - CSDN博客](https://blog.csdn.net/yangziluomu/article/details/52618173)

[^BP algorithm]: [反向传播算法](https://zh.wikipedia.org/wiki/%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD%E7%AE%97%E6%B3%95)
[^Moore-Penrose generalized inverse]: [摩尔－彭若斯广义逆](https://zh.wikipedia.org/wiki/%E6%91%A9%E5%B0%94%EF%BC%8D%E5%BD%AD%E8%8B%A5%E6%96%AF%E5%B9%BF%E4%B9%89%E9%80%86)

[^7]: [TagProp: Discriminative metric learning in nearest neighbor models for image auto-annotation](https://ieeexplore.ieee.org/document/5459266)
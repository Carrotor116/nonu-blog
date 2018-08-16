---
title: A NOVEL RATE CONTROL SCHEME FOR PANORAMIC VIDEO CODING (翻译)
date: 2018-08-15 23:40:24
tags: 论文翻译
---

## 摘要 (ABSTRACT) 
  多视角全景视频的留下被认为是VR内容生成的增加，由于它沉浸式的视觉体验。我们在本论文中要论证的是在评估压缩的全景视频的视觉质量上PSNR不如基于球面的PSNR(`S-PSNR`)来的有效，<!--more-->考虑后者全景视频的球面投影映射。因此，2D视频的编码的通常的速率控制(`RC`)模式，其主在与优化PSNR，是不适合于全景视频编码的。为了优化S-PSNR，我们在本论文中提出了一种新的全景视频编码RC模式。尤其是，我们发展了一个在约束比特率上的S-PSNR优化公式。然后，提出了一个开发公式的解，在全景视频编码中，这样的比特可以被收集到每个码块去优化S-PSNR。最后，实验结果证实了我们提出的RC模式在全景视频编码的S-PSNR改善的有效性。

## 1. 介绍 (INTRODUCTION)
  全景视频，作为VR内容的一种形式，在体验视频时提供360度的视觉方向。最近，全景视频由于其沉浸式的体验越来越流行。一方面，全景视频要求高分辨率(4K及以上)来提供良好的视频体验，如此巨大的交流带宽需要被其消耗。这会导致交流中的带宽饥饿问题。因此，有一个强烈的需求来提高全景视频压缩了。
  在过去的十年里，已经提出了几种方式[1-12]来最强全景视频的压缩效率。例如，在早期，提出的全景视频编码的用户驱动的交互压缩[1-3]。尤其是，全景视频被压缩到托基于多图块的流，并且一个具体的流集合被传输，编码和基于，根据用户当前的编码器视角。虽然近几年出现了一堆先进的用户驱动方式[4,5]，用户驱动方式在实践应用中仍然不成熟。这是因为在用户当前的视觉反馈上存在大量的传输延时。近来，提出了全景视频编码的编码器优化方式[6,7,11]。如，Zheng等人[6, 7]提出的一种新的运动补偿预测方式来提高全景视频的压缩质量。此外，Yu等人[11]提出的H.264投诉编码优化方式，自适应不同的多图块尺寸和比特率来提高整体的全景视频质量。相对与优化编码器，一些全景视频编码的预处理方式[8-10]被提出。在[8]中，Budagavi等人提出的在编码全景视频前利用区域自适应平滑，以至于更多的比特可以被保存在很少被看到的接近极端的领域中。相似的，Youvalari[10]提出的区域降采用预处理方式来减少由于紧张过度的极地地区引起的额外的比特率。可是，预处理方式不能控制全景视频编码的比特率，并且由于压缩全景视频很难以足够的比特率恢复原始质量。实际上，速率控制(RC)可以被应用于给定目标比特率下压缩全景视频的优化视觉质量。
  现存的二维视频编码标准，如最新的高效视频编码(HEVC)[13]，已经被使用在全景视频编码上，并且他们的RC模式可以被良好的应用。最近，根据HEVC提出的R-λ 模式[14]，作为最先进的RC模式。R-λ模式在速率失真优化和控制准确度方面表现良好。不幸的是，R-λ模式的RD优化主要关注于改善二维视频的PSNR。而对于全景视频编码，基于球面的PSNR(S-PSNR)[15]在测量视觉质量上更加有效。因此，按照优化S-PSNR的全景视频编码的比特率控制是需要的。按照我们所知的，全景视频编码的RC优化工作目前还没有。
  在本论文中，我们提出了一种全景视频编码的RC模式，通过在给定比特率下优化S-PSNR。实现，我们通过差异平均意见得分(DMOS)评估了一些压缩全景视频的主观质量。然后，我们发现相比PSNR，DMOS与S-PSNR更相关。因此，在我们的模式中采用S-PSNR来进行优化。于是，根据R-λ RC模式，我们提出了一个公式来在约束的目标比特率下优化S-PSNR。在我们的公式中，考虑了非一致S-PSNR的球面映射的采样。然后，我们开发了一个我们公式的解法，以至于比特按照S-PSNR被最优的分配到每个编码树单元(`CTU`)。总的来说，本论文的主要贡献如下：（1）我们证明了S-PSNR在评估全景视频编码中是有效的矩阵。（2）我们提出了在全景视频编码中优化S-PSNR的RC公式，通过一种能够达到最优CTU水平的比特分配的解决方法。
  
## 2. 初步 (PRELIMINARY)
  由于我们的RC模式主要致力于优化S-PSNR，我们将在2.1节简要回顾S-PSNR，并在2.2节验证它的有效性。

### 2.1 S-PSNR概要 (Overview of S-PSNR)
  S-PSNR主要在于测量给出参考视频的受损全景视频的失真(distortion)。与PSNR不同，S-PSNR是根据在球面区域的平均方差(MSE)，计算S-PSNR时需要这样的球面投影。图1显示了参考和受损序列的球面投影，公式中用于计算逐象素(pixel-wise)S-PSNR的平面区域像素是不平均采样的。
  {% asset_img fig_1.png Fig.1: 用于参考和修复序列的球面到平面投影的照明。这样的投影导致了在平面领域计算S-PSNR时采样的不均匀。  %}
  更加具体的说，我们假设球面上的像素代表`({g(n)}^N)n=1`，其中`N`是全景视频球面上的像素总数。并且，在球面上的g(n)是参考和受损序列上的投影位置`(xn，yn)`。给定`(xn，yn)`，S-PSNR可以根据平均球面平均方差(`S-MSE`)计算，如下。
{% asset_img formulate_1.png formulate_1.png  %}
其中`S(xn,yn)`和`S'(xn,yn)`是第n个像素投影到参考和受损序列上的像素值。然而，球面投影将`(xn,yn)`作为未知其像素值的自像素。为了解决这样的问题，根据他邻居像素的值，使用Lanczos插值法来评估`S(xn,yn)`和`S'(xn,yn)`。最后，S-PSNR可以通过公式1和`({(xn,yn)}^N)n=1`的插值获得，

### 2.2 验证S-PSNR (Verification of S-PSNR)
  为了验证S-PSNR的有效性，我们对比了PSNR和S-PSNR的表现，通过评估他们在几个全景视频序列中相关的主观质量。
  为了对比，在IEEE 1857 工作组[17]的标准测试集的原始格式中挑出8个全景视频序列。这些序列在500Kbps、3Mbps和100Mbps上被压缩。然后，对比参考原始序列，在压缩序列上测量PSNR和S-PSNR。 
  接下来，为了根据DMOS来量化那些压缩序列的主观主流，我们按照[18]的方法进行主观测试。在我们的主观测试中，有17个主体（包括10个男性和7个女性，年纪在19到30岁之间）参与评估了所有序列的的原始质量分数。这里，VR头戴式耳机Oculus Rift DK2被用来以原始分辨率(original resolution)播放测试全景视频。另外，我们利用LiveViewRife作为Oculus Rift视频播放，并采用它默认的球面投影类型。最后，在他们基础的评估主观得分上（rated subjective score），每个压缩序列的DMOS数值可以被计算出，如表1中呈现的。表1也显示了每个序列的PSNR和S-PSNR。  
  {% asset_img table_1.png table_1.png  %}
  现在，我们对每个压缩序列测量PSNR/S-PSNR和DMOS之间的相关性，使用皮尔逊相关系数(PCC)，斯皮尔曼等级顺序相关系数(SROCC),根MSE(RMSE)和平均绝对误差(MAE)。所有压缩序列的相关结果显示在表2中。观察表2，对比PSNR，S-PSNR与DMOS更相关。此外，DMOS和S-PSNR之间的错误比DMOS和PSNR之间的错误更小。因此，S-PSNR是比PSNR更合理的客观指标，证明了S-PSNR的有效性。  
  {% asset_img table_2.png table_2.png  %}

## 3. 提出率控制模式 (proposed rate control scheme) 
  RC在全景视频编码的主要目标是为了在给的的比特率上最大化S-PSNR，由于前述证实了S-PSNR的有效性。根据S-PSNR在公式(1)的定义,基于球面的失真是在球面上进行像素采样平方误差的总和(the sum of square error bettween pixels sampled from sphere)： 
  {% asset_img formulate_2.png  formulate_2.png  %}
  其中`Cm`是第m个`CTU`的像素集合。因此，在目标比特率R上，S-PSNR的优化公式可以如下：
  {% asset_img formulate_3.png  formulate_3.png  %}
  在公式3中，`rm`是第m个`CTU`的分配的比特数(the assigned bits)，并且`M`是当前帧下`CTUs`的总数。为了解上面的公式，需要引入一个拉格朗日乘数`λ`，并且公式3可以转化为一个无约束优化问题：
  {% asset_img formulate_4.png  formulate_4.png  %}
  这里，我们定义`J`作为RD成本的。通过设定公式4中的派生物(derivative) 为0，可以获得`J`的最小值：
  {% asset_img formulate_5.png  formulate_5.png  %}
  接下来，为了解公式5，我们需要模拟失真`dm`和比特率`rm`之间的关系。注意`dm`和`rm`是与`S-MSE`与每像素比特(bpp)除以一个CTU的像素和是各自等价的。和论文[14]相似，我们在四个编码全景视频的基础上，使用双曲线模型来探究球面失真`S-MSE`和比特率bpp。图2使用双曲线模型绘制了这四个序列的拟合RD曲线。在该图中，bpp可以通过公式6计算：
  {% asset_img formulate_6.png  formulate_6.png  %}
  {% asset_img fig_2.png R-D拟合曲线 %}
  其中`f`代表帧数，`W`和`H`分别代表视频的宽度和高度。图2显示，双曲线模型能够拟合S-MSE与bpp之间的关系，并且四个拟合曲线的R方(R-square)都大于0.99。因此，在我们的RC模式中以如下方式使用双曲线模型：
  {% asset_img formulate_7.png  formulate_7.png  %}
  其中`cm`和`ck`是双曲线模型的参数，可以通过使用论文[19]中的方式根据每个`CTU`来更新该参数。
  上面的等式可以重写为：
  {% asset_img formulate_8.png  formulate_8.png  %}
  根据公式5和公式8，如下等式成立：
  {% asset_img formulate_9.png  formulate_9.png  %}
  而且，根据公式3，我们可得如下约束：
  {% asset_img formulate_10.png formulate_10.png  %}
  在公式9和公式10上，每个`CTU`的比特分配可以按如下计算：
  {% asset_img formulate_11.png formulate_11.png  %}
  因此，一旦公式11能够解开，每个`CTU`中能使得`S-PSNR`最优的目标比特`rm`便可以得到。在本片论文中，我们应用最新的递归泰勒扩张(`RTE`)方法[19]来解公式11。值得注意的是，RTE方法的计算速度更快（每个CTU耗时0.0015ms），以至于我们的RC模式只需要比较少的计算开销。
  获得最优的比特率分配后，每个CTU的量化参数(`QP`)可以通过使用论文[14]中的方法来估得。图3总结了我们的全景视频编码RC模式的整个程序。注意，我们的RC模式主要适用于最新的根据HEVC的全景视频编码，并且可以通过重新调查比特率与失真的双曲线模型来扩张到其他的视频编码标准。
  {% asset_img fig_3.png fig_3.png  %}

## 4. 实验 (EXPERIMENT)
  在本节，采用试验来验证我们RC模式的有效性。4.1节展示了我们的实验用设置。4.2节，从RD表现、BD-rate和BD-PSNR方面评估我们的方法。4.3节讨论我们模式的准确性。

### 4.1 设置 (Settings)
  由于空间限制，我们的实验从IEEE 1857工作组[17]的测试集中选择了8个4K全景视频序列。他们如图4所示。这些序列都为10s时长30fps。图4显示了那些序列的内容，从室内到户外，包含人和和风景的场景。然后这些全景视频按HEVC参考软件HM-15.0来压缩。这里，我们在HM-15.0上实现我们的模式，然后将我们的模式与最新的R-λ RC模式[14]进行对比，后者是HM-15.0的默认RC设置。对于HM-15.0，低延迟P设置是通过配置文件`encoder_lowdelay_P_main.cfg`来配置的。与论文[14]中类似，我们使用惯例HM-15.0在四个固定的QPs（27，32，37和42）上压缩全景视频序列。然后，获得的比特率用于设置我们模式中和管理模式[14]中的每个序列的目标比特率。值得注意的是我们只对比最新的2D视频编码HEVC的RC模式[14]，因为目前没有全景视频编码的RC模式。
  {% asset_img fig_4.png  fig_4.png  %}
### 4.2 评估RD表现 (Evaluation on RD performance)
  **RD曲线**。我们使用y通道的S-PSNR来对比我们的模式与惯例模式[14]的RD表现，因为2.2节已经证实了在评估全景视频的主观质量上S-PSNR比PSNR更有效。我们在图5中绘制了所有测试全景视频的RD曲线，其中包含了我们的模式与惯例模式[14]。我们可以看出这些RD曲线中，在同样的比特率下我们的模式可以达到更高的S-PNSR。因此，在RD表现中，我们的模式更优秀。
  {% asset_img fig_5.png fig_5.png  %}
  **BD-PSNR和BD-rate**。接下来，我们在BD-PSNR和BD-rate上量化RD表现。与上面的RD曲线相似，我们使用y通道的S-PSNR来测量BD-PSNR和BD-rate。表4呈现了BD-PSNR上，我们的模式相对与管理[14]模式的改进。从表中可以看出，在BD-PSNR上我们模式平均改善了0.1613dB。这样的改善是主要因为我们的模式致力于优化S-PSNR，而[14]则在PSNR上优化。表4也列出了我们的模式和[14]的保留的BD-rate。我们可以看出与[14]对比，我们的模式平均能够挽回5.34%BD-rate。因此我们的模式在缓解全景视频的带宽饥饿问题上更有潜力。
  {% asset_img table_4.png table_4.png  %}
  **主观质量**。而且，在图6中显示了一个Dianynig序列的选中的帧视觉质量，该图是在同样比特率下使用HM-15.0编码我们的模式和惯例RC模式。我们可以观察到我们的模式区域的视觉质量比[14]的要好，更少的污点影响和更少的人工表现(less artifacts)。例如，我们模式产生的人物和灯光比[14]的更加清晰。除此之外，对比[14]，我们模式编码的腿部区域有更少的污点影响。总的来说，我们的模式比[14]表现得更好，通过对比RD曲线，BD-SPNR，BD-rate和主观质量。
  {% asset_img fig_6.png fig_6.png  %}

 ### 4.3 评估RC准确度 (Evaluation on RC accuracy)
  现在，我们来评估我们模式的RC准确度。在这项评估中，表4显示了我们模式和[14]模式的实际比特率关于目标比特率的错误率。我们可以从表中看出，对比[14]的错误率，平均RC错误率少了1%。此外，我们模式出现在tiyu_1的最大错误率3.02%，而[14]的错误率则高了1.37%。虽然我们模式的RC准确度比[14]的小，这仍然是很高的且与100%的准确度很接近。因此我们模式在控制根据HEVC的全景视频编码的控制上是有效且可行的。更重要的是，我们的RC能够提高全景视频的RC表现。
  {% asset_img table_4.png table_4.png  %}

## 5. 结论 (CONCLUSION)
  在本篇论文中，我们提出了一个新的全景视频编码RC模式，其减小了在目标比特率上根据球面的失真。尤其是，我们率先证实了相对于PSNR，最新的基于球面的失真测量矩阵S-PSNR的有效性。接下来，对于全景视频编码，相对于PSNR，我们的RC模式致力于优化S-PSNR。例如，通过S-PSNR评估，基于HEVC全景视频编码的编码有效性可以被提高，以至于有更好的主观质量。最后，实验结果表明我们的模式比最先进的R-λ RC模式要更好，平均0.16dB S-PSNR的改善和5.34% 的比特率挽回。

## 6. 引用 (REFERENCES)
[1] King-To Ng Shing-Chow Chan Heung-Yeung Shum "Data compression and transmission aspects of panoramic videos" IEEE Transactions on Circuits and Systems for Video Technology (TCSVT) vol. 15 no. 1 pp. 82-95 2005. 

[2] S. Heymann A. Smolic K. Mueller Y. Guo J. Rurainsky P. Eisert T. Wiegand "Representation coding and interactive rendering of high-resolution panoramic images and video using mpeg-4" Proc. Panoramic Photogrammetry Workshop (PPW) 2005. 

[3] Hideaki Kimata Shinya Shimizu Yutaka Kunita Megumi Isogai Yoshimitsu Ohtani "Panorama video coding for user-driven interactive video application" IEEE 13th InternationalSymposium on Consumer Electronics IEEE pp. 112-114 2009. 

[4] Gaddam Vamsidhar Reddy Michael Riegler Eg Ragnhild Carsten Gri-Wodz Pål Halvorsen "Tiling in interactive panoramic video: Approaches and evaluation" IEEE Transactions on Multimedia vol. 18 no. 9 pp. 1819-1831 2016. 

[5] Zare Alireza Aminlou Alireza M Hannuksela Miska Moncef Gabbouj "Hevc-compliant tile-based streaming of panoramic video for virtual reality applications" Proceedings of the ACM on Multimedia Conference (ACM MM) ACM pp. 601-605 2016. 

[6] Jiali Zheng Yanfei Shen Yongdong Zhang Ni Guangnan "Adaptive selection of motion models for panoramic video coding" IEEE International Conference on Multimedia and Expo (ICME) IEEE pp. 1319-1322 2007. 

[7] Zheng Jiali Zhang Yongdong Shen Yanfei Ni Guangnan "Panoramic video coding using affine motion compensated prediction" in Multimedia Content Analysis and Mining Springer pp. 112-121 2007. 

[8] Madhukar Budagavi John Furton Guoxin Jin Ankur Saxena Jeffrey Wilkinson Andrew Dickerson "360 degrees video coding using region adaptive smoothing" IEEE International Conference on Image Processing (ICIP). IEEE pp. 750-754 2015. 

[9] Jisheng Li Ziyu Wen Sihan Li Yikai Zhao Bichuan Guo Jiang-Tao Wen "Novel tile segmentation scheme for omnidirectional video" Image Processing (ICIP) 2016 IEEE International Conference on. IEEE pp. 370-374 2016. 

[10] Ramin Ghaznavi Youvalari 360-degree panoramic video coding 2016. 

[11] Matt Yu Haricharan Lakshman Bernd Girod "Content adaptive representations of omnidirectional videos for cinematic virtual reality" Proceedings of the 3rd International Workshop on Immersive Media Experiences ACM pp. 1-6 2015. 

[12] Ivana Tosic Pascal Frossard "Low bit-rate compression of omnidirectional images" Picture Coding Symposium IEEE pp. 1-4 2009. 

[13] Jens-Rainer Ohm Gary J Sullivan "High efficiency video coding: the next frontier in video compression [standards in a nutshell]" IEEE Signal Processing Magazine vol. 30 no. 1 pp. 152-158 2013. 

[14] Bin Li Houqiang Li Li Li Jinlei Zhang "Domain rate control algorithm for high efficiency video coding" IEEE Transactions on Image Processing (TIP) vol. 23 no. 9 pp. 3841-3854 2014. 

[15] Matt Yu Haricharan Lakshman Bernd Girod "A framework to evaluate omnidirectional video coding schemes" Mixed and Augmented Reality (ISMAR) 2015 IEEE InternationalSymposium on. IEEE pp. 31-36 2015. 

[16] Xiaoyu Xiu Yan Ye Yuwen He Bharath Vishwanath "Ahg8: Inter-digital's projection format conversion tool" Input Document to JVET 2016. 

[17] IEEE 1857 working group 2016 [online] Available: http://www.ieee1857.org. 

[18] Deng Xin Maosheng Bai Wei Wei "Draft of subjective evaluation methodology on vr video" Output Document of Standard for Immersive Visual Content Coding in IEEE 1857. 

[19] Shengxi Li Mai Xu Zulin Wang Xiaoyan Sun "Optimal bit allocation for ctu level rate control in hevc" IEEE Transactions on Circuits and Systems for Video Technology (TCSVT) 2016. 

> 原文  {% asset_link A_novel_rate_control_scheme_for_panoramic_video_coding.pdf  A NOVEL RATE CONTROL SCHEME FOR PANORAMIC VIDEO CODING %}
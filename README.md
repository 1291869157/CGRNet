## CGRNet (CVIU2024)
Scribble-based Complementary Graph Reasoning Network for Weakly Supervised Salient Object Detection

## Trained Model
Please download the trained model and put it in "models"

link: https://pan.baidu.com/s/1gKQS8Eua3M5kAs_XrMUQUQ  &nbsp;&nbsp;  code: 8nx6 

## Introduction
Current salient object detection (SOD) methods rely heavily on accurate pixel-level annotations. To reduce the annotation workload, some scribble-based methods have emerged. Recent works address the sparse scribble annotations by introducing auxiliary information and enhancing local features. However, the impact of long-range dependence between pixels on energy propagation and model performance has not been explored in this field. In this paper, we propose a novel complementary graph reasoning network (CGRNet), which globally infers relationships between salient regions by building graph representations. Specifically, we introduce a dual-stream cross-interactive graph reasoning pipeline to model high-level representations and incorporate efficient graph cooperation unit (GCU) to adaptively select complementary components from the representations. Additionally, considering the lack of structural information in scribble data, we design an edge-oriented module (EOM) to explicitly mine boundary semantics. Finally, we propose a dense fusion strategy (DFS) to aggregate multi-source semantics in a multi-guidance manner for obtaining complete global information. Experimental and visual results on five benchmarks demonstrate the superiority of our proposed CGRNet. 

## Overview
<div align="center">
  <img src="https://github.com/1291869157/CGRNet/blob/master/Overall.jpg" width="90%">
</div>

The CGRNet comprises three main components: graph inference network (GIN), edge oriented module (EOM) and multi-guidance saliency prediction module. Notably, our approach goes beyond the traditional method of solely extracting edge priors and salient features from the image to improve model performance. Instead, we use graph reasoning to capture long-range dependencies between pixels, enabling the network to transfer the information learned from scribble regions to distant unlabeled regions. This helps us recover the overall structure of salient objects from sparse scribble annotations.

## Our Results:
#### Visual Comparison
<div align="center">
  <img src="https://github.com/1291869157/CGRNet/blob/master/Fig5.jpg" width="90%">
</div>

#### Quantitative Comparison
<div align="center">
  <img src="https://github.com/1291869157/CGRNet/blob/master/fig3.jpg" width="90%">
</div>

We also provide the PR curve as below:
<div align="center">
  <img src="https://github.com/1291869157/CGRNet/blob/master/fig3.jpg" width="90%">
</div>

We provide saliency maps of our model on five benchmark saliency dataset (DUT, DUTS, ECSSD, HKU-IS, PASCAL-S) as below:
link: https://pan.baidu.com/s/1hn3hIqvmJMe30sch_wseCw &nbsp;   code: armt 


## Our Bib:
Please cite our paper if necessary:
```
@article{liang2024scribble,
  title={Scribble-based complementary graph reasoning network for weakly supervised salient object detection},
  author={Liang, Shuang and Yan, Zhiqi and Xie, Chi and Zhu, Hongming and Wang, Jiewen},
  journal={Computer Vision and Image Understanding},
  volume={243},
  pages={103977},
  year={2024},
  publisher={Elsevier}
}
```

# Contact
Please drop me an email for further problems or discussion: 1291869157@qq.com


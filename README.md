## CGRNet (CVIU2024)
Scribble-based Complementary Graph Reasoning Network for Weakly Supervised Salient Object Detection

## Trained Model
Please download the trained model and put it in "models"

Link: https://pan.baidu.com/s/1gKQS8Eua3M5kAs_XrMUQUQ  Code: 8nx6 

## Introduction
Current salient object detection (SOD) methods rely heavily on accurate pixel-level annotations. To reduce the annotation workload, some scribble-based methods have emerged. Recent works address the sparse scribble annotations by introducing auxiliary information and enhancing local features. However, the impact of long-range dependence between pixels on energy propagation and model performance has not been explored in this field. In this paper, we propose a novel complementary graph reasoning network (CGRNet), which globally infers relationships between salient regions by building graph representations. Specifically, we introduce a dual-stream cross-interactive graph reasoning pipeline to model high-level representations and incorporate efficient graph cooperation unit (GCU) to adaptively select complementary components from the representations. Additionally, considering the lack of structural information in scribble data, we design an edge-oriented module (EOM) to explicitly mine boundary semantics. Finally, we propose a dense fusion strategy (DFS) to aggregate multi-source semantics in a multi-guidance manner for obtaining complete global information. Experimental and visual results on five benchmarks demonstrate the superiority of our proposed CGRNet. 

## Overview


# Our Results:
![alt text](./results.png)

![alt text](./E_F_measure.png)

We provide saliency maps of our model on seven benchmark saliency dataset (DUT, DUTS, ECSSD, HKU-IS, PASCAL-S, SOD, THUR) as below:

https://drive.google.com/file/d/1njRCKDk89SX-um4aYN7vUV8ex05sI9ir/view?usp=sharing

# Benchmark Testing Dataset (DUT, DUTS, ECSSD, HKU-IS, PASCAL-S, SOD, THUR):

https://drive.google.com/open?id=11rPRBzqxdRz0zHYax995uvzQsZmTR4A7

# Our Bib:

Please cite our paper if necessary:
```
@inproceedings{jing2020weakly,
  title={Weakly-Supervised Salient Object Detection via Scribble Annotations},
  author={Zhang, Jing and Yu, Xin and Li, Aixuan and Song, Peipei and Liu, Bowen and Dai, Yuchao},
  booktitle=cvpr,
  year={2020}
}
```

# Contact

Please drop me an email for further problems or discussion: zjnwpu@gmail.com


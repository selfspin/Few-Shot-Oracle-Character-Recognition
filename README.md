# Few-Shot-Oracle-Character-Recognition
Fudan NNDL Final Project

文件组织形式

```
.
├── data
│   ├── dataset.py
│   ├── getclass.py
│   ├── Oracle-50K
│   ├── oracle_fs
│   └── oracle_source
├── output
│   └── 1_shot.jpg
├── test.py
├── train.py
├── try.py
└── visualize.py
```

## 运行

先运行`data/getclass.py`将每个字对应类别

再运行`train.py`

## 代码文件概述

`train.py`: ArcLoss版本，但是这个版本效果较差

`train_origin.py`: 最初最基本的版本，效果最好

`loss_functions.py`: 没用的loss function系列

`load.py`: 保存load之后的数据文件，先跑这个再跑 `train_origin.py`可以节省时间

`data/`: `dataset.py`: 最好的 dataloader

`data/`: `dataset_black.py`: 一个黑白照片的尝试，但是结果欠佳

`data/`: `dataset_masked.py`: 用了Bert之后的数据loader文件。

`data/`: `loading.py`: 用于加载数据文件的接口

`data/`: `FFD.py`: 没用的FFD变换

## 主要观点

* ImageNet 初始化有普遍的提升效果
* 

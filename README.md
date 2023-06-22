# TransNuSeg: A Lightweight Multi-Task Transformer for Nuclei Segmentation

This is the code for MICCAI2023 - TransNuSeg: A Lightweight Multi-Task Transformer for Nuclei Segmentation.

## Introduction
In this paper, we propose a lightweight multi-task framework for nuclei segmentation, namely TransNucSeg, as the first attempt at a fully Swin-Transformer driven architecture.  Innovatively, to alleviate the prediction inconsistency between branches, we propose a self-distillation loss that regulates the consistency between the nuclei decoder and normal edge decoder. And an innovative attention sharing scheme that shares attention heads amongst all decoders is employed to leverage the high correlation between tasks.

The overall architecture is demonstrated in the figure below. 

<p align="center">
  <img src="./model.png" />
</p>

## Dataset
In this paper, we test our model in microscopy and histology datasets.

The Fluorescence Microscopy Image Dataset is available [here](https://www.kaggle.com/hjh415/ca25net)

The Histology Image Dataset contains the combination of the open dataset [MoNuSeg](https://monuseg.grand-challenge.org/Data/) and another private histology dataset. 

## Quick Start
1, Download the datasets

2, Modify the hyperparameters, alpha, beta, gamma and sharing_ratio in [main.sh](./main.sh) or use the default value

3, Run main.sh

```bash
sh main.sh
```
Two folders named log and saved will be automatically created to store logging information and the trained model.


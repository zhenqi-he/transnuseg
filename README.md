# MICCAI 2023 - TransNuSeg: A Lightweight Multi-Task Transformer for Nuclei Segmentation

## Intro
In this paper, we propose a lightweight multi-task framework for nuclei segmentation. The overall archetecture is demonstrated in the figure below.

<p align="center">
  <img src="./model.png" />
</p>

## Dataset
In this paper, we test our model in microscopy and histology datasets.

The Radiology dataset can be downloaded [here](https://www.kaggle.com/hjh415/ca25net)

## Quick Start
1, Download the datasets

2, Modify the hyperparameters, alpha, beta, gamma and sharing_ratio in [main.sh](./main.sh) or use the default value

3, Run main.sh

```bash
sh main.sh
```
Two folders named log and saved will be automatically created to store logging information and the trained model.


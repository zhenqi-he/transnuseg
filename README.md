# TransNuSeg: A Lightweight Multi-Task Transformer for Nuclei Segmentation

This is the official implementation for **TransNuSeg: A Lightweight Multi-Task Transformer for Nuclei Segmentation** (MICCAI 2023). [paper](https://arxiv.org/pdf/2307.08051.pdf)

## Introduction
This paper proposes a lightweight multi-task framework for nuclei segmentation, namely TransNuSeg, as the first attempt at an entirely Swin-Transformer driven architecture.  Innovatively, to alleviate the prediction inconsistency between branches, we propose a self-distillation loss that regulates the consistency between the nuclei decoder and normal edge decoder. And an innovative attention-sharing scheme that shares attention heads amongst all decoders is employed to leverage the high correlation between tasks.

The overall architecture is demonstrated in the figure below. 

<p align="center">
  <img src="./model.jpg" />
</p>

## Dataset
In this paper, we test our model in Fluorescence Microscopy Image Dataset and Histology Image Dataset from [ClusterSeg](https://github.com/lu-yizhou/ClusterSeg). It is available [here](https://drive.google.com/drive/folders/1-ML_Z3yJOQsy3wbv__RL-qDZg7-n8-eI?usp=drive_link)


 

## Quick Start
1, Download the datasets from the above link and put them under the data folder.
```bash
cd data
unzip histology.zip
unzip fluorescence.zip
```

2, Modify the hyperparameters, alpha, beta, gamma and sharing_ratio in [main.sh](./main.sh) or use the default value

3, Run `main.sh` to the model

```bash
sh main.sh
```
Two folders named log and saved will be automatically created to store logging information and the trained model.


## Environment
The code is developed on one NVIDIA RTX 3090 GPU with 24 GB memory and tested in Python 3.8.10 and PyTorch 1.13.1.

## How to cite
You may cite us as
```
@InProceedings{transnuseg,
    author="He, Zhenqi
    and Unberath, Mathias
    and Ke, Jing
    and Shen, Yiqing",
    title="TransNuSeg: A Lightweight Multi-task Transformer for Nuclei Segmentation",
    booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI",
    year="2023",
    pages="206--215",
    isbn="978-3-031-43901-8"
}
```


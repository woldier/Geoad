<div align="center">

<h2 style="border-bottom: 1px solid lightgray;">
Geoad: Towards Geospatial Foundation Models via Efficient Continual Pre-Training
</h2>

<div style="display: flex; align-items: center; justify-content: center;">
<p align="center">
  <br align="center">
    <a href='#'><img src='http://img.shields.io/badge/Paper-arxiv.xxx.xxx-B31B1B.svg?logo=arXiv&logoColor=B31B1B'></a>
    <img alt="Static Badge" src="https://img.shields.io/badge/python-v3.8-green?logo=python">
    <img alt="Static Badge" src="https://img.shields.io/badge/torch-v2.1.2-B31B1B?logo=pytorch">
    <img alt="Static Badge" src="https://img.shields.io/badge/mmengine-v0.9.1-blue">
    <img alt="Static Badge" src="https://img.shields.io/badge/torchvision-v0.16.2-B31B1B?logo=pytorch">
    </br>
    <img alt="GitHub Issues or Pull Requests" src="https://img.shields.io/github/issues/woldier/Geoad">
    <img alt="GitHub Issues or Pull Requests" src="https://img.shields.io/github/issues-closed/woldier/Geoad?color=ab7df8">
    <img alt="GitHub forks" src="https://img.shields.io/github/forks/woldier/Geoad?style=flat&color=red">
    <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/woldier/Geoad?style=flat&color=af2626">
</p>
</div>
<br/>
<img src="figs/overview.png" alt="Geoad" style="max-width: 100%; height: auto;"/>
<div style="display: flex; align-items: center; justify-content: center;"> Network Overview </div>

</div>


### 🔍️🔍️ NEWS

- [2025/11/17] ✨✨ Init Repo.

[//]: # (- [2025/11/17] ✨✨ The [arxiv] paper will coming soon.)


### 📄📄 TODO

- ❎ submit to arxiv
- ❎ upload training code
- ❎ upload model weights

## Clone Repo

---
We add [mmpretrain](https://github.com/open-mmlab/mmpretrain),
[mmsegmentation](https://github.com/open-mmlab/mmsegmentation), and
[open-cd](https://github.com/likyoo/open-cd)
as our repository 
[submodule](https://git-scm.com/book/en/v2/Git-Tools-Submodules) .

So, one should clone this repository use the script as follows:

<details>
<summary>clone repository</summary>

```shell
git clone --recurse-submodules https://github.com/woldier/Geoad
```
> ### Tips
> If one already cloned the project and forgot --recurse-submodules,
> 
> ```shell
>  # cloned the project and forgot clone submodules 🥲🥲
>  git clone https://github.com/woldier/Geoad 
> 
>  # initialize and update each submodule in the repository 🥰🥰
>  git submodule update --init
>  ```
> 

</details>


after that, we link 

`submodules/mmpretrain/mmpretrain` $\to$ `mmpretrain`

`submodules/mmsegmentation/mmseg` $\to$ `mmseg`

`submodules/open-cd/opencd` $\to$ `opencd`
:

<details>
<summary>soft link</summary>

```shell
ln -s sumodules/mmpretrain/mmpretrain/ mmpretrain
ln -s sumodules/mmsegmentation/mmseg/ mmseg
ln -s sumodules/open-cd/opencd/ opencd
```
</details>


## 1. Creating Virtual Environment

---

This repo use `python-3.8`, for `nvcc -v` with `cuda >= 11.6`.

`torch 2.1.1`, `cuda 12.1`, `mmcv 2.1.0`, `mmengine 0.9.1`

<details>
<summary>Install script</summary>


```shell
conda create -n  peft-mmpretrain  python==3.8 -y
conda activate peft-mmpretrain


pip install torch==2.1.2+cu121  torchvision==0.16.2+cu121 -f https://download.pytorch.org/whl/torch_stable.html
# for CN user use follow script
pip install torch==2.1.2+cu121  torchvision==0.16.2+cu121 -f https://mirrors.aliyun.com/pytorch-wheels/cu121/  

pip install mmcv==2.1.0 mmengine==0.9.1 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html

pip install -r submodules/mmpretrain/mmpretrain/requirements/runtime.txt
pip install -r submodules/mmsegmentation/mmseg/requirements/runtime.txt
pip install -r submodules/open-cd/opencd/requirements/runtime.txt
```
</details>

Installation of the reference document refer:

Torch and torchvision versions relationship.

[![Official Repo](https://img.shields.io/badge/Pytorch-vision_refer-EE4C2C?logo=pytorch)](https://github.com/pytorch/vision#installation)
[![CSDN](https://img.shields.io/badge/CSDN-vision_refer-FC5531?logo=csdn)](https://blog.csdn.net/shiwanghualuo/article/details/122860521)

## 2. Initialization Checkpoints
Geoad relies on model weights from DINOv2 and MAE as an initialization to begin continual pre-training.
We have organized them here for your convenience.

| Model  |  ViT-L |
|--------|:---:|
| DINOv2 |  [ViT-L/14](https://download.openmmlab.com/mmpretrain/v1.0/dinov2/vit-large-p14_dinov2-pre_3rdparty_20230426-f3302d9e.pth) |
| MAE    |  [ViT-L/16](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-large-p16_8xb512-fp16-coslr-1600e_in1k/mae_vit-large-p16_8xb512-fp16-coslr-1600e_in1k_20220825-cc7e98c9.pth) |

## 3. Continual Pre-Training

### 3.1 Pre-training Dataset Preparation
Download the [MillionAID](https://jin-pu.github.io/Million-AID/) dataset. The data structure should be as follows:
```text
├── data/MillionAID
│   ├── train
│   │   ├── agriculture
│   │   │   ├── arable_land
│   │   │   │   ├── dry_field
│   │   │   │   │   ├── P0011232.jpg
│   │   │   │   │   ├── ...
│   │   │   │   ├── ...
│   │   │   ├── ...
│   │   ├── ...
│   ├── test
│   │   ├── xxx.jpg
│   │   ├── xxx.jpg
│   │   ├── ...
│   ├── ...
```
We use all the images in the training and test sets for pretraining.

### 3.2 Geoad Pre-training
To pre-train Geoad with multi-gpu distributed training, run the following on 1 nodes with 8 GPUs each:

<details>
<summary>pretraining script</summary>

```bash
# Geoad-DINOv2
tools/dist_train.sh config/geoad/frms_dinov2_vit-large-p14_8xb512-amp-coslr-100k_million-aid.py 8\
--launcher pytorch --auto-scale-lr

# Geoad-MAE
tools/dist_train.sh config/geoad/frms_mae_vit-large-p16_8xb128-amp-coslr-300e_million-aid.py 8\
--launcher pytorch --auto-scale-lr
```
</details>

## 4. Downstream Tasks

### 4.1 Sense Classification

### 4.2 Semantic Segmentation

### 4.3 Change Detection


## Acknowledgements
This project is built upon [OpenMMLab](https://openmmlab.com/codebase). We thank the OpenMMLab developers.

## Citation
If you use Geoad in your research, please cite:
```bibtex
@article{,
  title={},
  author={},
  journal={},
  year={}
}
```
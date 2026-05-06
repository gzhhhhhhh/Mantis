<h1 align="center">Mantis: Mamba-native Tuning is Efficient for 3D Point Cloud Foundation Models</h1>

<p align="center">
  Zihao Guo<sup>1</sup>, Jihua Zhu<sup>1*</sup>, Jian Liu<sup>2</sup>, Ajmal Saeed Mian<sup>3</sup>
</p>

<p align="center">
  <sup>1</sup> Xi’an Jiaotong University, Xi’an, China<br>
  <sup>2</sup> Singapore University of Technology and Design, Singapore<br>
  <sup>3</sup> University of Western Australia, Perth, Australia
</p>


<p align="center">
  <a href="https://arxiv.org/abs/2605.03438"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b.svg" alt="arXiv"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/Code%20License-Apache--2.0-green.svg" alt="Code License"></a>
</p>

<p align="center"><i>Mantis is a parameter-efficient fine-tuning framework for point cloud analysis built on Mamba backbones.</i></p>

## 📨 News

- [2026/04/27] Initial repository release.
- [2026/05/01] The official codebase and checkpoints are now publicly available.

## Abstract
Pre-trained 3D point cloud foundation models (PFMs) have demonstrated strong transferability across diverse downstream tasks. However, full fine-tuning these models is computationally expensive and storage-intensive. Parameter-efficient fine-tuning (PEFT) offers a promising alternative, but existing PEFT approaches are primarily designed for Transformer-based backbones and rely on token-level prompting or feature transformation. Mamba-based backbones introduce a granularity mismatch between token-level adaptation and state-level sequence dynamics. Consequently, straightforward transfer of existing PEFT approaches to frozen Mamba backbones leads to substantial accuracy degradation and unstable optimization. To address this issue, we propose Mantis, the first Mamba-native PEFT framework for 3D PFMs. Specifically, a State-Aware Adapter (SAA) is introduced to inject lightweight task-conditioned control signals into selective state-space updates, enabling state-level adaptation while keeping the pre-trained backbone frozen. Moreover, different valid point cloud serializations are regularized by Dual-Serialization Consistency Distillation (DSCD), thereby reducing serialization-induced instability. Extensive experiments across multiple benchmarks demonstrate that our Mantis achieves competitive performance with only about 5% trainable parameters.

## Overview

<div align="center">
  <img src="./media/pipeline.png" width="100%" alt="Mantis overview" />
</div>

## Getting Started
In the following, we will guide you how to use this repository step by step.🤗
### Requirements

- Python 3.10
- PyTorch 2.0.1
- CUDA 11.8
- GCC >= 4.9

### Quick Start

```bash
conda create -n mantis python=3.10 -y
conda activate mantis

# PyTorch
conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.8 -c pytorch -c nvidia

pip install -r requirements.txt

# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"

# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

# Chamfer Distance & emd
cd ./extensions/chamfer_dist
python setup.py install

cd ../emd
python setup.py install

# Mamba install
cd ../..
pip install causal-conv1d==1.1.1 mamba-ssm==1.1.1
```

### Datasets

Before running the code, please make sure the working directory is organized as follows:

<details>
<summary>click to expand 👈</summary>

```text
Mantis/
├── data/
│   ├── ModelNet/
│   │   └── modelnet40_normal_resampled/
│   ├── ModelNetFewshot/
│   │   ├── 5way_10shot/
│   │   ├── 5way_20shot/
│   │   ├── 10way_10shot/
│   │   └── 10way_20shot/
│   ├── ScanObjectNN/
│   │   ├── main_split/
│   │   └── main_split_nobg/
│   ├── ShapeNet55-34/
│   │   ├── shapenet_pc/
│   │   └── ShapeNet-55/
│   └── shapenetcore_partanno_segmentation_benchmark_v0_normal/
├── cfgs/
├── datasets/
└── ...
```
</details>

Here are the download links of the required datasets:

- `ShapeNet55 / ShapeNet34` (for pre-training): [link](https://drive.google.com/file/d/1jUB5yD7DP97-EqqU2A9mmr61JpNwZBVK/view?usp=sharing)
- `ScanObjectNN`: [link](https://hkust-vgd.github.io/scanobjectnn/)
- `ModelNet40`: [pre-processed](https://drive.google.com/drive/folders/1fAx8Jquh5ES92g1zm2WG6_ozgkwgHhUq?usp=sharing) or [raw](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip)
- `ModelNet Few-shot`: [link](https://drive.google.com/drive/folders/1gqvidcQsvdxP_3MdUr424Vkyjb_gt7TW?usp=sharing)
- `ShapeNetPart`: [link](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip)


## Main Results (Mamba3D)


| Task | Dataset  | Config | Acc. | Checkpoints Download |
| :-: | :-: | :-: | :-: | :-: |
| Pre-training | ShapeNet | N.A. | N.A. |  [Point-MAE](https://github.com/gzhhhhhhh/Mantis/releases/download/v1.0/pretrain_pointmae_ckpt-last.pth)|
| Classification | ScanObjectNN | [finetune_scan_objbg_mantis.yaml](./cfgs/finetune_scan_objbg_mantis.yaml) | 93.29% |  [OBJ_BG](https://github.com/gzhhhhhhh/Mantis/releases/download/v1.0/mantis_scan_objbg.pth)|
| Classification | ScanObjectNN  | [finetune_scan_objonly_mantis.yaml](./cfgs/finetune_scan_objonly_mantis.yaml) | 92.77% |[OBJ_ONLY](https://github.com/gzhhhhhhh/Mantis/releases/download/v1.0/mantis_scan_objonly.pth)| 
| Classification | ScanObjectNN  | [finetune_scan_hardest_mantis.yaml](./cfgs/finetune_scan_hardest_mantis.yaml) | 93.48% |[PB_T50_RS](https://github.com/gzhhhhhhh/Mantis/releases/download/v1.0/mantis_scan_hardest.pth)| 
| Classification | ModelNet40  | [finetune_modelnet_mantis.yaml](./cfgs/finetune_modelnet_mantis.yaml) | 94.70% |[ModelNet40](https://github.com/gzhhhhhhh/Mantis/releases/download/v1.0/mantis_modelnet.pth)| 
| Part segmentation | ShapeNetPart  | [partseg_mantis.yaml](./cfgs/partseg_mantis.yaml) | 86.10% mIoU |[Part_Seg](https://github.com/gzhhhhhhh/Mantis/releases/download/v1.0/mantis_part_seg.pth)| 

The evaluation commands with checkpoints should be in the following format:

```bash
CUDA_VISIBLE_DEVICES=<GPU> python main.py --test --config cfgs/finetune_scan_hardest_mantis.yaml --ckpts <path/to/ckpt> --exp_name <name>
```

## Fine-tuning on downstream tasks

### ModelNet40

```bash
# Fine-tune Mantis on ModelNet40.
CUDA_VISIBLE_DEVICES=<GPU> python main.py --config cfgs/finetune_modelnet_mantis.yaml --finetune_model --ckpts <path/to/pretrained_ckpt> --exp_name <name>
```

Although voting may further improve performance, we exclude it from the standard evaluation protocol because its additional test-time cost makes comparisons across different compute platforms less fair.


### ScanObjectNN

```bash
# Fine-tune Mantis on ScanObjectNN PB_T50_RS.
CUDA_VISIBLE_DEVICES=<GPU> python main.py --config cfgs/finetune_scan_hardest_mantis.yaml --finetune_model --ckpts <path/to/pretrained_ckpt> --exp_name <name>

# Fine-tune Mantis on ScanObjectNN OBJ_BG.
CUDA_VISIBLE_DEVICES=<GPU> python main.py --config cfgs/finetune_scan_objbg_mantis.yaml --finetune_model --ckpts <path/to/pretrained_ckpt> --exp_name <name>

# Fine-tune Mantis on ScanObjectNN OBJ_ONLY.
CUDA_VISIBLE_DEVICES=<GPU> python main.py --config cfgs/finetune_scan_objonly_mantis.yaml --finetune_model --ckpts <path/to/pretrained_ckpt> --exp_name <name>
```

### Part Segmentation

```bash
# Fine-tune Mantis on ShapeNetPart.
CUDA_VISIBLE_DEVICES=<GPU> python main.py --config cfgs/partseg_mantis.yaml --part_seg_model --ckpts <path/to/pretrained_ckpt> --exp_name <name>
```

## t-SNE visualization

```bash
CUDA_VISIBLE_DEVICES=<GPU> python main.py --tsne_model --config cfgs/finetune_scan_hardest_mantis.yaml --ckpts <path/to/ckpt> --test_model point_mae --tsne_fig_path tsne_mantis_scan_hardest.pdf --exp_name <name>
```

You may also define custom configurations for other visualization settings.

## Acknowledgements

This project is based on Mamba ([paper](https://arxiv.org/abs/2312.00752), [code](https://github.com/state-spaces/mamba)), Vision Mamba ([paper](https://arxiv.org/abs/2401.09417), [code](https://github.com/hustvl/Vim)), Point-MAE ([paper](https://arxiv.org/abs/2203.06604), [code](https://github.com/Pang-Yatian/Point-MAE)), and Mamba3D ([paper](https://arxiv.org/abs/2404.14966), [code](https://github.com/xhanxu/Mamba3D)). Thanks for their efforts.


## Citation

If you find this repository useful in your research, please consider giving a star ⭐ and a citation.

```
@misc{guo2026mantismambanativetuningefficient,
      title={Mantis: Mamba-native Tuning is Efficient for 3D Point Cloud Foundation Models}, 
      author={Zihao Guo and Jihua Zhu and Jian Liu and Ajmal Saeed Mian},
      year={2026},
      eprint={2605.03438},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.03438}, 
}
<div align = 'center'>
<h1>MDSAM</h1>
<h3>Multi-Scale and Detail-Enhanced Segment Anything Model for Salient Object Detection</h3>
Shixuan Gao, Pingping Zhang, Tianyu Yan, Huchuan Lu

Dalian University of Technology

Paper:([arxiv:2408.04326v1](https://arxiv.org/html/2408.04326v1))
</div>


## Abstract
Salient Object Detection (SOD) aims to identify and segment the most prominent objects in images. Advanced SOD methods often utilize various Convolutional Neural Networks (CNN) or Transformers for deep feature extraction. However, these methods still deliver low performance and poor generalization in complex cases. Recently, Segment Anything Model (SAM) has been proposed as a visual fundamental model, which gives strong segmentation and generalization capabilities. Nonetheless, SAM requires accurate prompts of target objects, which are unavailable in SOD. Additionally, SAM lacks the utilization of multi-scale and multi-level information, as well as the incorporation of fine-grained details. To address these shortcomings, we propose a Multi-scale and Detail-enhanced SAM (MDSAM) for SOD. Specifically, we first introduce a Lightweight Multi-Scale Adapter (LMSA), which allows SAM to learn multi-scale information with very few trainable parameters. Then, we propose a Multi-Level Fusion Module (MLFM) to comprehensively utilize the multi-level information from the SAM's encoder. Finally, we propose a Detail Enhancement Module (DEM) to incorporate SAM with fine-grained details. Experimental results demonstrate the superior performance of our model on multiple SOD datasets and its strong generalization on other segmentation tasks.

## Overview

<p align="center">
  <img src="assets/overall.png" alt="arch" width="80%">
</p>

## Usage

### Installation

#### Step 1:
Clone this repository

```bash
https://github.com/BellyBeauty/MDSAM.git
cd MDSAM
```

#### Step 2:

##### Create a new conda environment

```bash
conda create --name mdsam python=3.9
conda activate mdsam
```

##### Install Dependencies
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu<your cuda version>
pip install -r requirements.txt
```

##### Set up Datasets
````
-- SOD
    |-- DUTS
    |   |-- train
    |   |   |-- image
    |   |   |-- mask
    |   |-- test
    |   |   |-- image
    |   |   |-- mask
    |-- DUT-OMRON
    |   |-- test
    |   |   |-- image
    |   |   |-- mask
    |-- HKU-IS
    |   |-- test
    |   |   |-- image
    |   |   |-- mask
    |-- ECSSD
    |   |-- test
    |   |   |-- image
    |   |   |-- mask
    |-- PASCAL-S
    |   |-- test
    |   |   |-- image
    |   |   |-- mask
````
You can either go to [DUTS](http://saliencydetection.net/duts/), [DUT-OMRON](http://saliencydetection.net/dut-omron/), [HKU-IS](https://i.cs.hku.hk/~yzyu/research/deep_saliency.html), [ECSSD](https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html), and [PASCAL-S](https://cbs.ic.gatech.edu/salobj/) respectively to download and configure files, or you can directly download [SOD GoogleDrive](https://drive.google.com/file/d/1xY1nB1KMUNXYV0CyKTgYoYudekGFUp3R/view?usp=drive_link) that I have already configured.

##### Train

Download [ViT-B SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth), save it in ./ckpts

run ./scripts/train.sh

```bash
sh ./scripts/train.sh
```

example train.sh when you train from begining
```
CUDA_VISIBLE_DEVICES=<gpu device> python -m torch.distributed.launch --nproc_per_node=<gpu num> train.py \
    --batch_size 4 \
    --num_workers 4 \
    --data_path <path>/SOD/DUTS \
    --sam_ckpt <path>/sam_vit_b_01ec64.pth \
    --img_size 512
```

example train.sh when you load previous checkpoint
```
CUDA_VISIBLE_DEVICES=<gpu device> python -m torch.distributed.launch --nproc_per_node=<gpu num> train.py \
    --batch_size 4 \
    --num_workers 4 \
    --data_path <path>/SOD/DUTS \
    --resume <path>/model_epoch<epoch id>.pth
    --img_size 512
```

##### Evaluation

Download our pretrained checkpoint or train your own model!

| resolution | #params(M) | FLOPs(G) | FPS | download |
| :---: | :---: | :---: | :---: | :---: |
| 384 * 384 | 100 | 66 | 50 | - |
| 512 * 512 | 100 | 123 | 35| [GoogleDrive](https://drive.google.com/file/d/1jYow2ORHblp289zVneYRDTci29dTD8mf/view?usp=drive_link)

we use the evaluation code below to generate results:

[PySODMetrics: A simple and efficient implementation of SOD metrics](https://github.com/lartpang/PySODMetrics)

It should be noted that，we modify the calculation method of the max F-measure from selecting a unified highest threshold for all images to calculating the highest threshold for each individual image.

run ./scripts/eval.sh
```bash
sh ./scripts/eval.sh
```

example eval.sh
```
python evaluation.py \
    --data_path <path>/SOD \
    --img_size 512 \
    --checkpoint <path>/<checkpoint>.pth \
    --gpu_id 0 \
    --result_path <output>/<path>
```

## Predicted Saliency Map 

we provide the predicted saliency maps datasets DUTS-TE, DUT-OMRON, HKU-IS, ECSSD and PASCAL-S

| resolution | download |
| :---: | :---: |
| 384 * 384 | -
| 512 * 512 | [GoogleDrive](https://drive.google.com/file/d/1JM1riWxPMP2vVVAEl-Ah3HViO92RsoU5/view?usp=drive_link) |

## Citation

```
@article{gao2024multi,
  title={Multi-Scale and Detail-Enhanced Segment Anything Model for Salient Object Detection},
  author={Gao, Shixuan and Zhang, Pingping and Yan, Tianyu and Lu, Huchuan},
  journal={arXiv preprint arXiv:2408.04326},
  year={2024}
}
```

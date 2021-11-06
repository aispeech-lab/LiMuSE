# LiMuSE

## Overview

Pytorch implementation of our paper *LIMUSE: LIGHTWEIGHT MULTI-MODAL SPEAKER EXTRACTION*.

LiMuSE explores group communication on a multi-modal speaker extraction model and further compresses the model size with quantization strategy.

## Model

Our proposed model is a multi-steam architecture that takes multichannel mixture, target speaker’s enrolled utterance and visual sequences of detected faces as inputs, and outputs the target speaker’s mask in time domain. The encoded audio representations of mixture are then multiplied by the generated mask to obtain the target speech. Please see the figure below for detailed model structure.

![flowchart_limuse](https://github.com/aispeech-lab/LiMuSE/blob/main/images/flowchart_limuse.png)

## Datasets

We evaluate our system on two-speaker speech separation and speaker extraction problems using [GRID](https://pubmed.ncbi.nlm.nih.gov/17139705/) dataset. The pretrained face embedding extraction network is trained on [LRW](https://ieeexplore.ieee.org/document/8099850) dataset and [MS-Celeb-1M](https://www.researchgate.net/publication/305683616_MS-Celeb-1M_A_Dataset_and_Benchmark_for_Large-Scale_Face_Recognition) dataset. And we use [SMS-WSJ](https://arxiv.org/abs/1910.13934) toolkit to obtain simulated anechoic dual-channel audio mixture. We place 2 microphones at the center of the room. The distance between microphones is 7 cm.



## Getting Started

### Preparation

If you want to adjust configurations of the framework and the path of dataset, please modify the **option/train/train.yml** file.

### Training

Specify the path to train.yml file and run the training command:

```
python train.py -opt ./option/train/train.yml
```

This project supports full-precision and quantization training at the same time. Note that you need to modify two values of *QA_flag* in train.yml file if you would like to switch between full-precision and quantization stage.  *QA_flag* in training settings stands for weight quantization while the one in *net_conf* stands for activation quantization. 

### View tensorboardX

```
tensorboard --logdir ./tensorboard
```

## Result

- Hyperparameters of LiMuSE

  | Symbol | Description                                     | Value |
  | ------ | ----------------------------------------------- | ----- |
  | N      | Number of filters in auto-encoder               | 128   |
  | L      | Length of the filters (in audio samples)        | 16    |
  | T      | Temperature                                     | 5     |
  | X      | Number of GC-equipped TCN blocks in each repeat | 6     |
  | Ra     | Number of repeats in audio block                | 2     |
  | Rb     | Number of repeats in fusion block               | 1     |
  | K      | Number of groups                                | -     |

- Performance of LiMuSE and TasNet under various configurations. Q stands for quantization, VIS stands for visual cue and VP
  stands for voiceprint cue. Model size and compression ratio are also reported.

| Method                  |  K   | SI-SDR (dB) | #Params | Model    Size | Compression Ratio |
| :---------------------- | :--: | :---------: | :-----: | :-----------: | :---------------: |
| LiMuSE                  |  32  |    16.72    |  0.36M  |    0.16MB     |      223.75       |
|                         |  16  |    18.08    |  0.96M  |    0.40MB     |       89.50       |
| LiMuSE (w/o Q)          |  32  |    23.77    |  0.36M  |    1.44MB     |       24.86       |
|                         |  16  |    24.90    |  0.96M  |    3.84MB     |       9.32        |
| LiMuSE (w/o Q and VP)   |  32  |    18.60    |  0.19M  |    0.76MB     |       47.11       |
|                         |  16  |    24.20    |  0.52M  |    2.08MB     |       17.21       |
| LiMuSE (w/o Q and VIS)  |  32  |    15.68    |  0.22M  |    0.88MB     |       40.68       |
|                         |  16  |    21.91    |  0.55M  |    2.20MB     |       16.27       |
| LiMuSE (w/o Q and GC)   |  -   |    23.67    |  8.95M  |    35.8MB     |         1         |
| TasNet (dual-channel)   |  -   |    19.94    |  2.48M  |    9.92MB     |         -         |
| TasNet (single-channel) |  -   |    13.15    |  2.48M  |    9.92MB     |         -         |

## Citations

If you find this repo helpful, please consider citing:

```
@inproceedings{liu2021limuse,
  title={LIMUSE: LIGHTWEIGHT MULTI-MODAL SPEAKER EXTRACTION},
  author={Liu, Qinghua and Huang, Yating and Hao, Yunzhe and Xu, Jiaming and Xu, Bo},
  booktitle={ArXiv},
  year={2021},
}
```

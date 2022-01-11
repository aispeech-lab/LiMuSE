# LiMuSE

## Overview

Pytorch implementation of our paper [LIMUSE: LIGHTWEIGHT MULTI-MODAL SPEAKER EXTRACTION](https://arxiv.org/abs/2111.04063).

LiMuSE explores group communication on a multi-modal speaker extraction model and further compresses the model size with quantization strategy.

## Model

Our proposed model is a multi-steam architecture that takes multichannel mixture, target speaker’s enrolled utterance and visual sequences of detected faces as inputs, and outputs the target speaker’s mask in time domain. The encoded audio representations of mixture are then multiplied by the generated mask to obtain the target speech. Please see the figure below for detailed model structure.

![flowchart_limuse](https://github.com/aispeech-lab/LiMuSE/blob/main/images/flowchart_limuse.png)

## Datasets

We evaluate our system on two-speaker speech separation and speaker extraction problems using [GRID](http://spandh.dcs.shef.ac.uk/gridcorpus) dataset. The pretrained face embedding extraction network is trained on [LRW](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html) dataset and [MS-Celeb-1M](https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities-real-world/) dataset. And we use [SMS-WSJ](https://github.com/fgnt/sms_wsj) toolkit to obtain simulated anechoic dual-channel audio mixture. We place 2 microphones at the center of the room. The distance between microphones is 7 cm.



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

**Table 1:** Performance of LiMuSE and TasNet under various configurations. Q stands for quantization, VIS stands for visual cue and VP stands for voiceprint cue. Model size and MACs are also reported.

| Method                   |  K   | SI-SDR (dB) | #Params  (M) | Model  Size (MB) | MACs (G) |
| :----------------------- | :--: | :---------: | :----------: | :--------------: | :------: |
| LiMuSE                   |  32  |    16.72    |     0.36     |       0.16       |  11.60   |
|                          |  16  |    18.08    |     0.96     |       0.40       |  23.26   |
| LiMuSE  (w/o Q)          |  64  |    18.39    |     0.29     |       1.16       |   6.45   |
|                          |  32  |    23.77    |     0.36     |       1.44       |  11.60   |
|                          |  16  |    24.90    |     0.96     |       3.84       |  23.26   |
| LiMuSE  (w/o Q and VP)   |  32  |    18.60    |     0.19     |       0.76       |   6.76   |
|                          |  16  |    24.20    |     0.52     |       2.08       |  13.14   |
| LiMuSE  (w/o Q and VIS)  |  32  |    15.68    |     0.22     |       0.88       |   6.76   |
|                          |  16  |    21.91    |     0.55     |       2.20       |  13.13   |
| LiMuSE  (w/o Q and GC)   |  -   |    23.67    |     8.95     |      35.80       |  53.35   |
| AVMS                     |  -   |    13.72    |     4.90     |      19.60       |  29.46   |
| TasNet  (dual-channel)   |  -   |    19.94    |     2.48     |       9.92       |  15.20   |
|                          |  32  |    21.08    |     0.10     |       0.72       |   3.91   |
|                          |  16  |    23.13    |     0.26     |       1.16       |   7.09   |
| TasNet  (single-channel) |  -   |    13.15    |     2.48     |       9.92       |  15.19   |

## Citations

If you find this repo helpful, please consider citing:

```
@inproceedings{liu2021limuse,
  title={LIMUSE: LIGHTWEIGHT MULTI-MODAL SPEAKER EXTRACTION},
  author={Liu, Qinghua and Huang, Yating and Hao, Yunzhe and Xu, Jiaming and Xu, Bo},
  booktitle={arXiv:2111.04063},
  year={2021},
}
```

# LiMuSE

## Overview

📰 **Paper [[link](https://arxiv.org/abs/2111.04063)]**

📄**Code [[link](https://github.com/aispeech-lab/LiMuSE)]**

📄**Dataset [[link](https://academictorrents.com/details/3cd18ff2d3eec881207dcc5ca5a2c3a2a3afe462)]**

PyTorch implementation of our paper [LIMUSE: LIGHTWEIGHT MULTI-MODAL SPEAKER EXTRACTION](https://arxiv.org/abs/2111.04063).

- In this paper, we propose a lightweight multi-modal speaker extraction framework, which incorporates multi-channel information, target speaker's visual feature and voiceprint as reference information and further apply Group Communication, Context Codec and an ultra-low bit quantization technology to reduce the model size and complexity while maintaining relatively high performance.


- What's more, we release source code and the dataset including extracted features used in our experiments to help you get started on our project quickly. Feel free to contact us if you have any questions or suggestions.


## Model

Our proposed model is a multi-stream architecture that takes multi-channel audio mixture, target speaker’s enrolled utterance and visual sequences of detected faces as inputs, and outputs the target speaker’s mask in the time domain. The encoded audio representations of the mixture are then multiplied by the generated mask to obtain the target audio. Please see the figure below for detailed model structure.

![flowchart](https://github.com/aispeech-lab/LiMuSE/blob/main/images/flowchart.png)

## Datasets

We evaluate our system on two-speaker speech separation and speaker extraction problems using [GRID](https://pubmed.ncbi.nlm.nih.gov/17139705/) dataset. The pretrained face embedding extraction network is trained on [LRW](https://ieeexplore.ieee.org/document/8099850) dataset and [MS-Celeb-1M](https://www.researchgate.net/publication/305683616_MS-Celeb-1M_A_Dataset_and_Benchmark_for_Large-Scale_Face_Recognition) dataset. And we use [SMS-WSJ](https://arxiv.org/abs/1910.13934) toolkit to obtain simulated anechoic dual-channel audio mixture. We place 2 microphones at the center of the room. The distance between microphones is 7 cm.

## Getting Started

### Requirements

- PyTorch version >= 1.6.0
- Python version >= 3.6

### Preparing Dataset

We have uploaded the dataset and extracted visual feature and speaker embeddings from video sequences and reference audios ahead of time so that you can directly download the dataset we released [here](https://academictorrents.com/details/3cd18ff2d3eec881207dcc5ca5a2c3a2a3afe462) and go on to the next step.

The directories are arranged like this:

```
data
├── lip_fea
|	├── test
|	├── train
|	├── valid
├── mixture
|	├── test
|	├── train
|	├── valid
├── ref
|	├── test
|	├── train
|	├── valid
├── target
|	├── test
|	├── train
|	├── valid
├── grid_vp.pkl
```

### Configuration

If you want to adjust configurations of the framework and the path of dataset, please modify the configuration file in **option/train/train.yml**.

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
  | N      | Number of channels in audio encoder             | 128   |
  | L      | Length of the filters (in audio samples)        | 32    |
  | P      | Kernel size in convolutional blocks             | 3     |
  | Ra   | Number of repeats in audio block                | 2     |
  | Rf   | Number of repeats in fusion block               | 1     |
  | C      |  Context size (in frames)                       | 32    |
  | K      |  Number of groups                               | -     |
  | Wq   |  Weight Quantization bit                        | 3     |
  | Aq   |  Activation Quantization bit                    | 8     |
  | T0   |  Temperature increment per epoch                | 5     |

- Performance of LiMuSE under various configurations and comparison with baselines. K stands for number of groups. Q stands for quantization, CC stands for Context Codec, VIS stands for visual cue and VP stands for voiceprint cue. 1ch and 2ch represents single-channel and dual-channel mix speech input stream.

| Method                 | K  | SI-SDR | SDR   | SDRi  | #Param | Model Size    | MACs           |
| ---------------------- | -- | ------ | ----- | ----- | ------ | ------------- | -------------- |
| LiMuSE                 | 32 | 15.53  | 16.67 | 16.46 | 0.41M  | 0.19MB(0.56%) | 3.98G  (7.46%) |
|                        | 16 | 17.25  | 17.71 | 17.50 | 1.12M  | 0.48MB(1.41%) | 7.52G  (14.1%) |
| LiMuSE (w/o Q)         | 32 | 21.75  | 22.61 | 22.40 | 0.41M  | 0.19MB(0.56%) | 3.98G  (7.46%) |
|                        | 16 | 24.27  | 24.83 | 24.63 | 1.12M  | 0.48MB(1.41%) | 7.52G  (14.1%) |
| LiMuSE (w/o Q and CC)  | 32 | 19.17  | 20.71 | 20.50 | 0.37M  | 1.40MB(4.1%)  | 5.94G (11.1%)  |
|                        | 16 | 23.78  | 23.65 | 23.45 | 0.97M  | 3.70MB(10.8%) | 11.77G (22.1%) |
| LiMuSE (w/o Q and VP)  | 32 | 20.66  | 21.73 | 21.53 | 0.21M  | 0.81MB(2.37%) | 2.25G (4.22%)  |
|                        | 16 | 21.13  | 22.38 | 22.17 | 0.60M  | 2.30MB(6.47%) | 4.03G (7.56%)  |
| LiMuSE (w/o Q and VIS) | 32 | 14.75  | 16.32 | 16.11 | 0.25M  | 0.94MB(2.75%) | 2.25G (4.22%)  |
|                        | 16 | 18.57  | 20.75 | 20.54 | 0.63M  | 2.42MB(7.09%) | 4.03G (7.56%)  |
| LiMuSE (raw 2ch)       | -  | 23.54  | 24.02 | 23.83 | 8.95M  | 34.14MB(100%) | 53.34G (100%)  |
| LiMuSE (raw 1ch)       | -  | 12.43  | 13.37 | 13.16 | 8.95M  | 34.13MB       | 53.33G         |
| AVMS                   | -  | -      | -     | 15.74 | 5.80M  | 22.34MB       | 60.66G         |
| AVDC                   | -  | -      | 9.30  | 8.88  | -      | -             | -              |
| Conv-TasNet            | -  | 14.97  | 15.48 | 15.27 | 3.48M  | 13.28MB       | 21.44G         |

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

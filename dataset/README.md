# Multi_Channel_Grid

## Overview

Here we release the dataset (Multi_Channel_Grid, abbreviated as **MC_Grid**) used in our paper [LIMUSE: LIGHTWEIGHT MULTI-MODAL SPEAKER EXTRACTION](https://arxiv.org/abs/2111.04063).

MC_Grid, which is based on [GRID](http://spandh.dcs.shef.ac.uk/gridcorpus/) dataset, includes multi-channel audio, extracted voiceprint and visual feature. The method of feature extraction will be introduced below.

MC_Grid is specially prepared for speaker extraction task, and our code is available at [aispeech-lab/LiMuSE](https://github.com/aispeech-lab/LiMuSE). Feel free to contact us if you have any questions or suggestions.

## File Structure

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

Also, we provide data list file for you, in which each line represents a different training or testing sample.

For example, 

```
mixture/test/0_RIR0_s16_lgak6a_s12_pbbb6a_mix.wav target/test/0_RIR0_s16_lgak6a_target.wav ref/test/s16/swwn7n.wav lip_fea/test/s16/lgak6a.npy 5.033008905777031 3.927016329225026 4.42914342632976 3.025639973625415 f m
```

From left to right are dual-channel audio mixture, target audio, reference audio, visual feature, target azimuth, target distance, interfering azimuth, interfering distance, target gender and interfering gender.

## Multi-Channel audio mixture generation

[GRID](http://spandh.dcs.shef.ac.uk/gridcorpus/) dataset contains 18 male speakers and 15 female speakers, and each of them has 1,000 frontal 3 second long face video recordings. We randomly select 3 males and 3 females to construct a validation set of 2.5 hours and another 3 males and 3 females for a test set of 2.5 hours. The rest of the speakers form the training set of 30 hours. To construct a 2-speaker mixture, we randomly choose two different speakers and select audio from each chosen speaker, then mix these two clips at a random SNR level between -5 dB and 5 dB.

We use [SMS-WSJ](https://arxiv.org/abs/1910.13934) toolkit to obtain simulated anechoic dual-channel audio mixture. We place 2 microphones at the center of the room. The distance between microphones is 7 cm.

## Voiceprint

We use an off-the-shelf speaker diarization toolkit [pyannote](https://github.com/pyannote/pyannote-audio) to extract the speaker embeddings. You can also use your own methods to process reference audio to obtain speaker embeddings.

## Visual Feature

We obtain face embeddings of the target speaker in videos using the network in [DAVS](https://arxiv.org/abs/1807.07860), which extracts speech-related visual features from visual inputs explicitly by the adversarially disentangled method. The pretrained face embedding extraction network is trained on [LRW](https://ieeexplore.ieee.org/document/8099850) dataset and [MS-Celeb-1M](https://www.researchgate.net/publication/305683616_MS-Celeb-1M_A_Dataset_and_Benchmark_for_Large-Scale_Face_Recognition) dataset.

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

```
@article{cooke2006audio,
  title={An audio-visual corpus for speech perception and automatic speech recognition},
  author={Cooke, Martin and Barker, Jon and Cunningham, Stuart and Shao, Xu},
  journal={The Journal of the Acoustical Society of America},
  volume={120},
  number={5},
  pages={2421--2424},
  year={2006},
  publisher={Acoustical Society of America}
}
```

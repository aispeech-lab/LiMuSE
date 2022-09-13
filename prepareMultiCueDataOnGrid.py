# -*- coding: utf-8 -*-
'''
Prepare multi-cue data of Grid.
'''

__all__ = [
    "PrepareMultiCueGridDataSamples",
]


import json
import os
import soundfile as sf
import random
import numpy as np
from scipy import signal
import yaml
import utils
from random import random as rand
import resampy
import pickle
import copy
# from pyannote.core import SlidingWindow
import torch
from prepareMultiCueDataForGeneral import PrepareMultiCueDataSamples


class PrepareMultiCueGridDataSamples(PrepareMultiCueDataSamples):

    def __init__(self, config):
        super(PrepareMultiCueGridDataSamples, self).__init__(config)
        # print('Initialize PrepareMultiCueGridDataSamples...')
        with open(config.VP_PKL_PATH, 'rb') as fi:
            self.vp_dict = pickle.load(fi)
        self.previous_batch_num = 0

    def get_samples(self, phase='train'):
        if phase == 'train':
            sample_lst, sample_num = self.train_lst, self.train_num
            batch_size = self.config.BATCH_SIZE
            previous_batch_num = self.previous_batch_num
        elif phase == 'valid':
            sample_lst, sample_num = self.valid_lst, self.valid_num
            sample_num = min(200, sample_num)
            batch_size = self.config.BATCH_SIZE
            previous_batch_num = 0
        elif phase == 'test':
            sample_lst, sample_num = self.test_lst, self.test_num
            batch_size = self.config.TEST_BATCH_SIZE
            previous_batch_num = 0
        batch_num = int(sample_num / batch_size)
        # evaluation_interval = min(5000, batch_num-previous_batch_num) if phase == 'train' else batch_num  
        evaluation_interval = batch_num
        last_batch_num = previous_batch_num + evaluation_interval
        for batch_idx in range(previous_batch_num, last_batch_num + 1):
            if batch_idx == last_batch_num:
                if phase == 'train':
                    self.previous_batch_num = batch_idx if batch_idx < batch_num else 0
                yield False

            # print('\n')
            mix_wav_lst = list()
            spk1_wav_lst = list()
            gt_spk1_visual_lst = list()
            gt_spk1_ref_vp_lst = list()

            available_cue_lst = self.cues
            infer_cue_dict = {'mask': [], 'interfere': [], 'replace': []}
            if phase == 'train':
                if self.cue_missing_training:
                    available_cue_lst = self.get_available_cue_lst(self.cues)
                if self.cue_infering_training:
                    infer_cue_dict = self.get_infer_cue_dict()         

            for batch_sample_idx in range(batch_size):
                sample_idx = batch_idx * batch_size + batch_sample_idx
                sensors_file, spk1_file, spk1_ref_file, spk1_visual_file, azimuth, target_spk_dis, interfering_spk_azimuth, interfering_spk_dis, target_spk_gender, interfering_spk_gender = sample_lst[
                    sample_idx]
                sensors_file = os.path.join(self.sample_root_path, sensors_file)
                spk1_file = os.path.join(self.sample_root_path, spk1_file)
                spk1_ref_file = os.path.join(self.sample_root_path, spk1_ref_file)
                spk1_visual_file = os.path.join(self.sample_root_path, spk1_visual_file)
                mix_wav, wav_rate = sf.read(sensors_file)
                mix_wav = mix_wav.transpose()  # channel * T
                if not 'binaural' in available_cue_lst:
                    mix_wav[1] = mix_wav[0]
                    azimuth = 0
                spk1_wav, wav_rate = sf.read(spk1_file)
                spk1_wav = spk1_wav.transpose()[0]

                spk1_ref_wav, wav_rate = sf.read(spk1_ref_file)
                spk1_ref_wav = spk1_ref_wav.transpose()
                spk1_ref_wav = self.preprocess_data(spk1_ref_wav, wav_rate)
                spk1_ref_vp = self.vp_dict['_'.join(spk1_file.split('/')[-1].split('_')[2:4])+'.wav']  # 目标纯净语音声纹做GT声纹

                try:
                    spk1_visual = np.load(spk1_visual_file)
                except FileNotFoundError as e:
                    # print(e)
                    spk1_visual = np.zeros((75, 256))
                    visual_valid = 0
                else:
                    visual_valid = 1
                    
                azimuth = float(azimuth)
                target_spk_dis = float(target_spk_dis)
                interfering_spk_azimuth = float(interfering_spk_azimuth)
                interfering_spk_dis = float(interfering_spk_dis)
 
                gt_azimuth, gt_spk1_visual, gt_spk1_ref_vp, gt_spk1_ref_wav = copy.deepcopy(azimuth), copy.deepcopy(spk1_visual), copy.deepcopy(spk1_ref_vp), copy.deepcopy(spk1_ref_wav)

                mix_wav_lst.append(mix_wav)
                spk1_wav_lst.append(spk1_wav)                
                gt_spk1_visual_lst.append(gt_spk1_visual)
                gt_spk1_ref_vp_lst.append(gt_spk1_ref_vp)

            mix_wav_lst = torch.tensor(mix_wav_lst, dtype=torch.float)                 
            spk1_wav_lst = torch.tensor(spk1_wav_lst, dtype=torch.float)
            gt_spk1_ref_vp_lst = torch.tensor(gt_spk1_ref_vp_lst, dtype=torch.float)
            gt_spk1_visual_lst = torch.tensor(gt_spk1_visual_lst, dtype=torch.float)
            yield {'mix_wav': mix_wav_lst,
                   'spk1_wav': spk1_wav_lst,
                   'gt_visual': gt_spk1_visual_lst,
                   'gt_ref_vp': gt_spk1_ref_vp_lst}

if __name__ == "__main__":
    config = utils.read_config('/mnt/lustre/xushuang4/liuqinghua/LiMuSE/options/train/train.yml')
    grid_samples = PrepareMultiCueGridDataSamples(config)
    
    epoch = 0
    while epoch < 5:
        epoch += 1
        step = 0
        grid_samples_iterator = grid_samples.get_samples(phase='test')
        while True:
            test_data = grid_samples_iterator.__next__()
            print(test_data['mix_wav'])
            if test_data == False:
                break
            print('step:{}', step)
            step += 1

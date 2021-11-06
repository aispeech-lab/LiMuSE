# <encoding:utf8>
"""
# by jacoxu 20200728
"""

from numpy.core.arrayprint import dtype_short_repr
import torch
import os
import numpy as np
from pyannote.core import Segment, SlidingWindow, SlidingWindowFeature
from scipy.spatial.distance import cdist
import time
import math
import pickle


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%5dm %2ds' % (m, s)

# ls -R | awk '{print i$0}' i=`pwd`  # to learn
if __name__ == '__main__':
    root_list = ['/ssd2/dataset/grid/RIR/2spk/test/stage0_speech_zeromean/resaudio_test/',
                 '/ssd2/dataset/grid/RIR/2spk/train/stage0_speech_zeromean/resaudio_train/',
                 '/ssd2/dataset/grid/RIR/2spk/valid/stage0_speech_zeromean/resaudio_valid/']
    data_file = 'grid_spk_sentence.txt'
    data_list = open(data_file, 'r', encoding='utf-8')
    
    torch.set_grad_enabled(False)
    model = torch.hub.load('pyannote/pyannote-audio', 'emb')
    model.chunks_ = SlidingWindow(duration=2, step=0.5)
    print(f'Embedding has dimension {model.dimension:d}.')
    grid_vp_dict = {}
    for line in data_list.readlines():
        wav_path = line.strip()
        _, mode_spk_sentence = wav_path.split('resaudio_')
        mode, spk, sentence = mode_spk_sentence.split('/')
        fea_key = '_'.join([spk, sentence])
        spk_embed = model({'audio': wav_path})
        spk_embed = np.mean(spk_embed, axis=0, keepdims=True)
        print('fea_key: {spk_embed}:', {fea_key: spk_embed})
        print('spk_embed.shape:', spk_embed.shape)
        grid_vp_dict.update({fea_key: spk_embed})
    fo = open('./grid_vp.pkl', 'wb')
    pickle.dump(grid_vp_dict, fo)


    # root = '/ssd1/dataset/AVSpeech/clips'
    # video_root = '/ssd1/cuijian/voice_verify/tmp/clips'
    # # 加载声纹特征提取模型
    # torch.set_grad_enabled(False)
    # model = torch.hub.load('pyannote/pyannote-audio', 'emb')
    # # model = hub.load('pyannote/pyannote-audio', 'emb')  # .list('pytorch/vision', force_reload=False)
    # model.chunks_ = SlidingWindow(duration=2, step=0.5)
    # print(f'Embedding has dimension {model.dimension:d}.')

    # check_file = open('filter_file_voicecheck.txt', "a+", encoding='utf-8')
    # with open('filter_file_voicecheck.txt', 'r') as f:  # 打开文件
    #     lines = f.readlines()  # 读取所有行
    #     last_line = lines[-1]  # 取最后一行
    #     # 处理到的文件
    # begin_str = last_line.split(',')[0]  # 例如'xan/CYiKiwKmCEc'
    # begin_folder = begin_str.split('/')[0]  # 例如'xan'

    # # root_video_file = os.listdir(root)
    # # root_video_file.sort()

    # folder_socket = True
    # socket = True  # 控制何时开始

    # nonmatch_num = 0
    # total_num = 0
    # remain_file = open('filter_file.txt', "r", encoding='utf-8')
    # start_time = time.time()
    # for line in remain_file.readlines():
    #     spk_folder_path = line.strip('\n')  # 去掉列表中每一个元素的换行符
    #     if spk_folder_path == begin_str:
    #         folder_socket = False
    #         continue
    #     if folder_socket:
    #         continue
    #     # 如果文件不存在，则解压
    #     spk_upper_folder = spk_folder_path.split('/')[0]
    #     if not os.path.exists(os.path.join(video_root, spk_upper_folder)):
    #         os.system("tar -xf %s/%s.tar -C %s"
    #                   % (root, spk_upper_folder, video_root))
    #     root_video_path = os.path.join(video_root, spk_upper_folder)
    #     # 如果不存在此压缩文件，则跳出（由于xkw无法正常解压）
    #     if not os.path.exists(root_video_path):
    #         continue
    #     spk_folder = os.path.join(video_root, spk_folder_path)

    #     clip_video_file_list = os.listdir(spk_folder)
    #     clip_video_file_list.sort()
    #     for clip_video_file in clip_video_file_list:
    #         if clip_video_file[-3:] != 'mp4':
    #             clip_video_file_list.remove(clip_video_file)

    #     first_sub_video_path = os.path.join(spk_folder, clip_video_file_list[0])
    #     first_sub_wav_path = str(first_sub_video_path)[:-4] + '.wav'
    #     # -- 是为了避免输出文件名以-开头的问题
    #     if not os.path.exists(first_sub_wav_path):
    #         os.system("ffmpeg -i %s -f wav -ar 16000 -- %s -loglevel quiet" % (first_sub_video_path, first_sub_wav_path))

    #     # 取第一个clip的声纹做为校验声纹
    #     spk_embed = model({'audio': first_sub_wav_path})

    #     is_valid = False
    #     distance = 1.0
    #     for sub_video in clip_video_file_list[1:]:
    #         sub_video_path = os.path.join(spk_folder, sub_video)
    #         sub_wav_path = str(sub_video_path)[:-4] + '.wav'
    #         if not os.path.exists(sub_wav_path):
    #             os.system("ffmpeg -i %s -f wav -ar 16000 -- %s -loglevel quiet" % (sub_video_path, sub_wav_path))

    #         spk_embed_check = model({'audio': sub_wav_path})

    #         distance = cdist(np.mean(spk_embed, axis=0, keepdims=True),
    #                          np.mean(spk_embed_check, axis=0, keepdims=True),
    #                          metric='cosine')[0, 0]

    #         if distance > 0.5:
    #             is_valid = False
    #             break
    #         else:
    #             is_valid = True

    #     total_num += 1
    #     if is_valid:
    #         check_file.write('%s, True, %f\n' % (spk_folder_path, distance))
    #     else:
    #         nonmatch_num += 1
    #         check_file.write('%s, False, %f\n' % (spk_folder_path, distance))
    #     check_file.flush()

    #     print('####, Current Path:%s, Total num:%08d, Cost Time:%s' % (
    #         spk_folder_path, total_num, timeSince(start_time)))

    #     os.system('rm -r %s' % (video_root+'/'+spk_folder_path))

    # remain_file.close()
    # check_file.close()

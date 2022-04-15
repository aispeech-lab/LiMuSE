# dummy = "resaudio_test_"

# def openreadtxt(file_name):
#     data = []
#     file = open(file_name,'r')  #打开文件
#     file_data = file.readlines() #读取所有行
#     for row in file_data:
#         tmp_list = row.split(' ') #按‘，’切分每行的数据
#         mixture = tmp_list[0].split(dummy)
#         mix_path = mixture[0]+mixture[1]+mixture[2].replace('.wav','_mix.wav')
#         data.append('mixture/'+mix_path.split("grid_")[1])
        
#         target = tmp_list[1].split(dummy)
#         target_path = target[0]+target[1]+'target.wav'
#         data.append('target/'+target_path.split("grid_")[1])
        
#         ref = tmp_list[3].split("resaudio_")
#         data.append('ref/'+ref[1])
#         data.append(tmp_list[5])
#         data.append(tmp_list[7])
#         data.append(tmp_list[8])
#         data.append(tmp_list[9])
#         data.append(tmp_list[10])
#         data.append(tmp_list[11])
#         data.append(tmp_list[12])
#     return data
 

# if __name__=="__main__":
#     data = openreadtxt('mix_2_spk_voice_multi_channel_test.txt')
#     # print('data:', data)
#     file_handle=open('datalist_2mix_test.txt',mode='w')
#     for i in data:
#         if "\n" in i:
#             file_handle.writelines(i)
#         else:
#             file_handle.writelines(i+' ')
#     file_handle.close()
#     print('Done writing')

import os

# path = '/mnt/lustre/xushuang4/dataset/grid_to_release/mixture/test'
# path_list = os.listdir(path)
# for str in path_list:
#     # print(str)
#     out = str.split("resaudio_test_")
#     adjust = out[0]+out[1]+out[2].replace('.wav', '_mix.wav')
#     # print(adjust)
#     # print(str)
#     os.rename(path+'/'+str, path+'/'+adjust)

# print('done')

path = '/mnt/lustre/xushuang4/dataset/grid_to_release/target/test'
path_list = os.listdir(path)
for str in path_list:
    # if '_S1' in str:
    #     os.remove(path+'/'+str)
    # print(str)
    out = str.split("resaudio_test_")
    adjust = out[0]+out[1]+'target.wav'
    os.rename(path+'/'+str, path+'/'+adjust)

print('done')
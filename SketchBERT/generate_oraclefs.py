import torch
import numpy as np
import os
import pickle as pkl
import torch
from PIL import Image

f = open('../Oracle/data/oracle_fs/seq/char_to_idx.txt', encoding='utf-8')
cls = f.read()
f.close()

for shot in [1, 3, 5]:

    sum_path = '../Oracle/data/oracle_fs/seq/oracle_200_{}_shot/'.format(shot)
    # a file with all npz names in quickdraw, a line with a path.
    modes = ['train', 'valid', 'test']
    offsets = {'train': {}, 'valid': {}, 'test': {}}
    save_dir = 'Oracle_fs/oracle_200_{}_shot'.format(shot)  # simple_memmap
    load_file = {'full': open('Oracle_fs/oracle_200_{}_shot/memmap_sum.txt'.format(shot), 'w')}
    max_len = 0

    file_list = os.listdir(sum_path)

    for line in file_list:
        data_path = sum_path + line
        # data_path = data_path.replace('/npz/','/npz/') #simple_npz
        if not os.path.exists(data_path):
            continue
        data = np.load(data_path, encoding='latin1', allow_pickle=True)
        tmp_data = []
        tmp_num = 0
        for mode in modes:
            tmp_data = []
            load_mode = 'train'
            for sketch in data[load_mode]:
                tmp_data.append(sketch)
            cls_name = cls[int(line[0:-4])]
            save_path = os.path.join(save_dir, '{}_{}.dat'.format(cls_name, mode))
            offsets[mode][cls_name] = []
            start = 0
            max_len = 0
            len_record = []
            for sketch in tmp_data:
                if len(sketch.shape) != 2 or sketch.shape[1] != 3:
                    print(sketch)
                    continue
                end = start + sketch.shape[0]
                len_record.append(sketch.shape[0])
                # print(sketch.shape)
                max_len = max(max_len, sketch.shape[0])
                offsets[mode][cls_name].append((start, end))
                start = end
            len_record = np.array(len_record)
            tmp_num += len(tmp_data)
            print(mode, 'mean:{}, std:{}, max:{}, min:{}'.format(len_record.mean(), len_record.std(), len_record.max(),
                                                                 len_record.min()))
            max_len = 250
            print('Num Count: < {} :{}, total:{}'.format(max_len, (len_record < max_len).sum(), len(len_record)))
            # print(save_path)

            stack_data = np.concatenate(tmp_data, axis=0)
            # print(stack_data.dtype, stack_data)
            tmp_memmap = np.memmap(save_path, dtype=np.int16, mode='write', shape=stack_data.shape)
            tmp_memmap[:] = stack_data[:]
            tmp_memmap.flush()
        load_file['full'].write('{}\t{}\n'.format(os.path.join(save_dir, '{}.dat'.format(cls_name)), tmp_num))


    pkl.dump(offsets, open('Oracle_fs/oracle_200_{}_shot/offsets.npz'.format(shot), 'wb'))

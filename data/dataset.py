from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import torch


class OracleFS(Dataset):
    def __init__(self, dataset_type, shot=1):
        """
        dataset_type: ['train', 'test']
        """
        self.root = 'data/oracle_fs/img/oracle_200_{shot}_shot/{type}'.format(shot=shot, type=dataset_type)
        self.shot = shot
        self.class_to_oracle = np.load('data/oracle_fs/img/class_to_oracle.npy', allow_pickle=True).item()
        self.oracle_to_class = np.load('data/oracle_fs/img/oracle_to_class.npy', allow_pickle=True).item()
        self.data = []
        for oracle in self.oracle_to_class.keys():
            path = os.path.join(self.root, oracle)
            files = os.listdir(path)
            for file in files:
                if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
                    img = Image.open(os.path.join(path, file))
                    img = np.array(img.convert('RGB'))
                    img = img.transpose((2, 0, 1))
                    self.data.append([torch.tensor(img).to(torch.float32), self.oracle_to_class[oracle]])

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    shot = 1
    dataset_type = 'train'
    root = 'oracle_fs/img/oracle_200_{shot}_shot/{type}'.format(shot=shot, type=dataset_type)
    class_to_oracle = np.load('oracle_fs/img/class_to_oracle.npy', allow_pickle=True).item()
    oracle_to_class = np.load('oracle_fs/img/oracle_to_class.npy', allow_pickle=True).item()
    data = []
    for oracle in oracle_to_class.keys():
        path = os.path.join(root, oracle)
        files = os.listdir(path)
        for file in files:
            if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
                img = Image.open(os.path.join(path, file))
                data.append([np.array(img), oracle_to_class[oracle]])

    data = 0

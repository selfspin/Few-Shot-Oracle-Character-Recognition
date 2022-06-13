import torchvision.transforms
from PIL import Image
from torch.utils.data import Dataset
from data.FFD import ffd
import numpy as np
import os
import torch
import torchvision.transforms as transforms
import re

basic_transfrom = transforms.Compose([
    transforms.Resize((244, 244)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float),
])


class OracleFS_Masked(Dataset):
    def __init__(self, shot=1, transform=basic_transfrom, enlarge=3, Normalize=False):
        """
        dataset_type: ['train', 'test']
        """
        self.root = 'data/oracle_fs/img/oracle_200_{shot}_shot/{type}'.format(shot=shot, type='train')
        self.root_masked = 'data/Generate_oracle_fs/oracle_200_{shot}_shot'.format(shot=shot)
        self.shot = shot
        self.class_to_oracle = np.load('data/oracle_fs/img/class_to_oracle.npy', allow_pickle=True).item()
        self.oracle_to_class = np.load('data/oracle_fs/img/oracle_to_class.npy', allow_pickle=True).item()
        self.data = []
        self.transform = transform
        self.enlarge = enlarge

        for oracle in self.oracle_to_class.keys():
            # basic imgs
            path = os.path.join(self.root, oracle)
            files = os.listdir(path)
            for file in files:
                if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
                    img = Image.open(os.path.join(path, file)).convert('RGB')
                    for _ in range(self.enlarge):
                        self.data.append([img, self.oracle_to_class[oracle]])
                    # if enlarge > 0 and self.transform is not None:
                    #     for _ in range(self.enlarge):
                    #         tr_img = self.transform(img)
                    #         self.data.append([tr_img, self.oracle_to_class[oracle]])

            # masked process image
            path = os.path.join(self.root_masked, oracle)
            files = os.listdir(path)
            for file in files:
                if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
                    img = Image.open(os.path.join(path, file)).convert('RGB')
                    for _ in range(self.enlarge):
                        self.data.append([img, self.oracle_to_class[oracle]])
                    # if enlarge > 0 and self.transform is not None:
                    #     for _ in range(self.enlarge):
                    #         tr_img = self.transform(img)
            #         #         self.data.append([tr_img, self.oracle_to_class[oracle]])

    def __getitem__(self, index):
        img, label = self.transform(self.data[index][0]), self.data[index][1]
        return img, label

    def __len__(self):
        return len(self.data)


def get_mean_std(train_type, data):
    """
    :return: the mean and standard deviation
    """
    num_imgs = len(data)
    means = np.zeros(3)
    stdevs = np.zeros(3)

    for i in range(num_imgs):
        means += torch.mean(data[i][0], dim=[1, 2]).numpy()
        stdevs += torch.sum(torch.square(data[i][0]), dim=[1, 2]).numpy()
    means /= num_imgs
    std = np.sqrt((stdevs - (num_imgs * 244 * 244) * means ** 2) / (num_imgs * 244 * 244 - 1))

    print("{} : normMean = {}".format(train_type, means))
    print("{} : normstdevs = {}".format(train_type, std))

    return means, std


if __name__ == "__main__":
    root_masked = 'Generate_oracle_fs/oracle_200_{shot}_shot'.format(shot=1)
    class_to_oracle = np.load('oracle_fs/img/class_to_oracle.npy', allow_pickle=True).item()
    oracle_to_class = np.load('oracle_fs/img/oracle_to_class.npy', allow_pickle=True).item()



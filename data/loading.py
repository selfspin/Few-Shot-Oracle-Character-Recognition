import torchvision.transforms
from PIL import Image
from torch.utils.data import Dataset
from data.FFD import ffd
import numpy as np
import os
import torch
import torchvision.transforms as transforms
import pickle

basic_transfrom = transforms.Compose([
    transforms.Resize((244, 244)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float),
])


class OracleFS_train(Dataset):
    def __init__(self, shot=1, transform=basic_transfrom, enlarge=2):
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
        self.enlarge = enlarge - 1
        num = 1
        for oracle in self.oracle_to_class.keys():
            path = os.path.join(self.root, oracle)
            files = os.listdir(path)
            for file in files:
                if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
                    img = Image.open(os.path.join(path, file)).convert('RGB')
                    self.data.append([basic_transfrom(img), self.oracle_to_class[oracle]])
                    if enlarge > 0 and self.transform is not None:
                        for _ in range(self.enlarge):
                            tr_img = self.transform(img)
                            self.data.append([tr_img, self.oracle_to_class[oracle]])

            # # masked process image
            # path = os.path.join(self.root_masked, oracle)
            # files = os.listdir(path)
            # for file in files:
            #     if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
            #         img = Image.open(os.path.join(path, file)).convert('RGB')
            #         self.data.append([basic_transfrom(img), self.oracle_to_class[oracle]])
            #         if enlarge > 0 and self.transform is not None:
            #             for _ in range(self.enlarge):
            #                 tr_img = self.transform(img)
            #                 self.data.append([tr_img, self.oracle_to_class[oracle]])

            print(f"{num} | already loading {oracle}")
            num = num + 1

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1]

    def __len__(self):
        return len(self.data)


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(244, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
    transforms.RandomHorizontalFlip(p=1),
    # transforms.RandomVerticalFlip(p=0.5),
    transforms.RandAugment(num_ops=2, magnitude=12),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    # Cutout(),
    torchvision.transforms.ToTensor(),
])

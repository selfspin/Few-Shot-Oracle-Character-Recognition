# import sys, io
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='surrogatepass')

import os
import random

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
from data.loading import *
import random
import pickle

device = 'cuda'
seed_value = 0
np.random.seed(seed_value)
torch.manual_seed(seed_value)
random.seed(seed_value)
if device != 'cpu':
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(244, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomVerticalFlip(p=0.5),
    transforms.RandAugment(num_ops=2, magnitude=12),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    # Cutout(),
    transforms.ToTensor(),
])

exp_transform = transforms.Compose([
    transforms.RandomResizedCrop(244, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomVerticalFlip(p=0.5),
    transforms.RandAugment(num_ops=2, magnitude=12),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    # Cutout(),
    # transforms.ToTensor(),
])

transform_list = [
    transforms.Compose([
        transforms.RandomResizedCrop(244, scale=(0.7, 0.9), ratio=(1.0, 1.0)),
    ]),
    transforms.Compose([
        transforms.Resize(244),
        transforms.RandomHorizontalFlip(p=1),
    ]),
    transforms.Compose([
        transforms.Resize(244),
        transforms.RandAugment(num_ops=2, magnitude=12),
    ]),
    transforms.Compose([
            transforms.Resize(244),
    transforms.ColorJitter(0.2, 0.2, 0.2)]),
]

basic_transfrom = transforms.Compose([
    transforms.Resize((244, 244)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float),
])

trainset = OracleFS_combined(shot=1, transform=train_transform, basic_enlarge=10, aug_enlarge=1)
with open('data/1_shot_train_combine.pickle', 'wb') as f:
    pickle.dump(trainset, f)

trainset = OracleFS_combined(shot=3, transform=train_transform, basic_enlarge=10, aug_enlarge=1)
with open('data/3_shot_train_combine.pickle', 'wb') as f:
    pickle.dump(trainset, f)

trainset = OracleFS_combined(shot=5, transform=train_transform, basic_enlarge=10, aug_enlarge=1)
with open('data/5_shot_train_combine.pickle', 'wb') as f:
    pickle.dump(trainset, f)


if __name__ == '__main__':
    root = 'data/oracle_fs/img/oracle_200_{shot}_shot/{type}'.format(shot=1, type='train')
    root_masked = 'data/Generate_oracle_fs/oracle_200_{shot}_shot'.format(shot=1)
    shot = 1
    class_to_oracle = np.load('data/oracle_fs/img/class_to_oracle.npy', allow_pickle=True).item()
    oracle_to_class = np.load('data/oracle_fs/img/oracle_to_class.npy', allow_pickle=True).item()
    data = []

    enlarge = 12
    num = 1
    transform = exp_transform
    fig, axes = plt.subplots(3, 4, figsize=(4, 3))
    fig_m, axes_m = plt.subplots(3, 4, figsize=(4, 3))

    for oracle in oracle_to_class.keys():
        path = os.path.join(root, oracle)
        files = os.listdir(path)
        for file in files:
            if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
                img = Image.open(os.path.join(path, file)).convert('RGB')
                axes_m[2, 3].imshow(img)
                axes_m[2, 3].set_axis_off()
                for i in range(12):
                    tr_img = np.array(transform(img))
                    x, y = int(np.floor(i / 3)), i % 3
                    axes[y, x].imshow(tr_img[:, :, ::-1].astype(int))
                    axes[y, x].set_axis_off()

        iter = 0
        path = os.path.join(root_masked, oracle)
        files = os.listdir(path)
        for file in files:
            if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
                img = Image.open(os.path.join(path, file)).convert('RGB')
                tr_img = np.array(transform(img))
                x, y = int(np.floor(iter / 3)), iter % 3
                axes_m[y, x].imshow(tr_img[:, :, ::-1].astype(int))
                axes_m[y, x].set_axis_off()
                iter = iter + 1
        break

# trainset = OracleFS_best(shot=1, transform=train_transform, enlarge=20)
# with open('data/5_shot_train.pickle', 'wb') as f:
#     pickle.dump(trainset, f)

import torchvision.transforms
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image
import torch

basic_transfrom = torchvision.transforms.Resize((244, 244))


class OracleFS(Dataset):
    def __init__(self, dataset_type, shot=1, transform=basic_transfrom, enlarge=2, Normalize=False):
        """
        dataset_type: ['train', 'test']
        """
        self.root = 'data/oracle_fs/img/oracle_200_{shot}_shot/{type}'.format(shot=shot, type=dataset_type)
        self.type = dataset_type
        self.shot = shot
        self.class_to_oracle = np.load('data/oracle_fs/img/class_to_oracle.npy', allow_pickle=True).item()
        self.oracle_to_class = np.load('data/oracle_fs/img/oracle_to_class.npy', allow_pickle=True).item()
        self.data = []
        self.transform = transform
        self.enlarge = enlarge
        self.Normalize = Normalize

        for oracle in self.oracle_to_class.keys():
            path = os.path.join(self.root, oracle)
            files = os.listdir(path)
            for file in files:
                if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
                    img = Image.open(os.path.join(path, file))
                    img = np.array(img.convert('RGB'))
                    img = img.transpose((2, 0, 1))
                    img = torch.tensor(img).to(torch.float32)
                    if self.type == 'test':
                        self.data.append([self.transform(img), self.oracle_to_class[oracle]])
                    if self.type == 'train':
                        self.data.append([basic_transfrom(img), self.oracle_to_class[oracle]])
                        if enlarge > 0 and self.transform is not None:
                            for _ in range(enlarge):
                                self.data.append([basic_transfrom(img), self.oracle_to_class[oracle]])

        if self.Normalize:
            m, s = self.get_mean_std()
            norm = torchvision.transforms.Normalize(m, s)
            for img in self.data:
                img = norm(img)

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1]

    def __len__(self):
        return len(self.data)

    def get_mean_std(self):
        """
        :return: the mean and standard deviation
        """
        means = np.zeros(3)
        stdevs = np.zeros(3)
        num_imgs = len(self.data)
        for img in self.data:
            for i in range(3):
                # 一个通道的均值和标准差
                means[i] += img[i, :, :].mean().item()
                stdevs[i] += img[i, :, :].std().item()

        means /= len(self.data)
        stdevs /= len(self.data)

        print("{} : normMean = {}".format(self.type, self.means))
        print("{} : normstdevs = {}".format(self.type, self.stdevs))

        return means, stdevs


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

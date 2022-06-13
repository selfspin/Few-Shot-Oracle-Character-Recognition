import torchvision.transforms
from PIL import Image
from torch.utils.data import Dataset
from data.FFD import ffd
import numpy as np
import os
import torch
import torchvision.transforms as transforms

basic_transfrom = transforms.Compose([
    transforms.Resize((244, 244)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float),
])


class OracleFS(Dataset):
    def __init__(self, dataset_type, shot=1, transform=basic_transfrom, enlarge=3, Normalize=False):
        """
        dataset_type: ['train', 'test']
        """
        # self.root = 'oracle_fs/img/oracle_200_{shot}_shot/{type}'.format(shot=shot, type=dataset_type)
        self.root = 'data/oracle_fs/img/oracle_200_{shot}_shot/{type}'.format(shot=shot, type=dataset_type)
        self.type = dataset_type
        self.shot = shot
        # self.class_to_oracle = np.load('oracle_fs/img/class_to_oracle.npy', allow_pickle=True).item()
        # self.oracle_to_class = np.load('oracle_fs/img/oracle_to_class.npy', allow_pickle=True).item()
        self.class_to_oracle = np.load('data/oracle_fs/img/class_to_oracle.npy', allow_pickle=True).item()
        self.oracle_to_class = np.load('data/oracle_fs/img/oracle_to_class.npy', allow_pickle=True).item()
        self.data = []
        self.transform = transform
        self.enlarge = enlarge - 1
        self.Normalize = Normalize

        for oracle in self.oracle_to_class.keys():
            path = os.path.join(self.root, oracle)
            files = os.listdir(path)
            for file in files:
                if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
                    if self.type == 'test':
                        img = Image.open(os.path.join(path, file))
                        self.data.append([self.transform(img), self.oracle_to_class[oracle]])
                    if self.type == 'train':
                        img = Image.open(os.path.join(path, file))
                        self.data.append([basic_transfrom(img), self.oracle_to_class[oracle]])
                        if enlarge > 0 and self.transform is not None:
                            for _ in range(self.enlarge):
                                # tr_img = self.transform(ffd(img).convert('RGB'))  # basic_transfrom(ffd(img).convert('RGB'))
                                tr_img = self.transform(img)  # \
                                # if np.random.rand() < 1 else \
                                # self.transform(ffd(img).convert('RGB'))  # basic_transfrom(ffd(img).convert('RGB'))
                                self.data.append([tr_img, self.oracle_to_class[oracle]])

        if self.Normalize:
            m, s = get_mean_std(self.type, self.data)
            norm = transforms.Compose([torchvision.transforms.Normalize(m, s),
                                       # transforms.RandomErasing(p=0.2)
                                       ])
            for i in range(len(self.data)):
                self.data[i][0] = norm(self.data[i][0])

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1]

    def __len__(self):
        return len(self.data)


class SiameseNetworkDataset(Dataset):

    def __init__(self, dataset_type, shot=1, transform=basic_transfrom, enlarge=3):
        super().__init__(dataset_type, shot, transform, enlarge)
        # self.root = 'oracle_fs/img/oracle_200_{shot}_shot/{type}'.format(shot=shot, type=dataset_type)
        self.root = 'data/oracle_fs/img/oracle_200_{shot}_shot/{type}'.format(shot=shot, type=dataset_type)
        self.template = 'data/oracle_fs/img/oracle_200_{shot}_shot/{type}'.format(shot=shot, type=dataset_type)
        self.type = dataset_type
        self.shot = shot
        # self.class_to_oracle = np.load('oracle_fs/img/class_to_oracle.npy', allow_pickle=True).item()
        # self.oracle_to_class = np.load('oracle_fs/img/oracle_to_class.npy', allow_pickle=True).item()
        self.class_to_oracle = np.load('data/oracle_fs/img/class_to_oracle.npy', allow_pickle=True).item()
        self.oracle_to_class = np.load('data/oracle_fs/img/oracle_to_class.npy', allow_pickle=True).item()
        self.data = []
        self.transform = transform
        self.enlarge = enlarge - 1

        for oracle in self.oracle_to_class.keys():
            path = os.path.join(self.root, oracle)
            files = os.listdir(path)
            for file in files:
                if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
                    img = Image.open(os.path.join(path, file)).convert('RGB')
                    self.data.append([basic_transfrom(img), self.oracle_to_class[oracle]])

        self.training_df = []

        if self.type == 'train':
            for img0, class0 in self.data:
                for img1, class1 in self.data:
                    if enlarge > 0:
                        for _ in range(enlarge):
                            self.training_df.append([self.transform(img0),
                                                     self.transform(img1),
                                                     class0 == class1])

    # Apply image transformations
    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1]

    def __len__(self):
        return len(self.training_df)


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


if __name__ == '__main__':
    # shot = 1
    # dataset_type = 'train'
    # root = 'oracle_fs/img/oracle_200_{shot}_shot/{type}'.format(shot=shot, type=dataset_type)
    # class_to_oracle = np.load('oracle_fs/img/class_to_oracle.npy', allow_pickle=True).item()
    # oracle_to_class = np.load('oracle_fs/img/oracle_to_class.npy', allow_pickle=True).item()
    # data = []
    # for oracle in oracle_to_class.keys():
    #     path = os.path.join(root, oracle)
    #     files = os.listdir(path)
    #     for file in files:
    #         if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
    #             img = Image.open(os.path.join(path, file))
    #             data.append([np.array(img), oracle_to_class[oracle]])
    #
    # data = 0
    train_transform = transforms.Compose([
        transforms.Resize(244),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.ConvertImageDtype(torch.float),
    ])

    print("Loading training data ...")
    trainset = OracleFS(dataset_type='train', shot=1, transform=train_transform)

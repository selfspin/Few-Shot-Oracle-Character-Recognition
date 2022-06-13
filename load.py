import torch
from torchvision import transforms
import pickle
from data.loading import OracleFS_best

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(244, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomVerticalFlip(p=0.5),
    transforms.RandAugment(num_ops=2, magnitude=12),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    # Cutout(),
    transforms.ToTensor(),
])

basic_transfrom = transforms.Compose([
    transforms.Resize((244, 244)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float),
])


trainset = OracleFS_best(shot=1, transform=train_transform, enlarge=20)
with open('data/1_shot_train.pickle', 'wb') as f:
    pickle.dump(trainset, f)

trainset = OracleFS_best(shot=1, transform=train_transform, enlarge=20)
with open('data/3_shot_train.pickle', 'wb') as f:
    pickle.dump(trainset, f)

trainset = OracleFS_best(shot=1, transform=train_transform, enlarge=20)
with open('data/5_shot_train.pickle', 'wb') as f:
    pickle.dump(trainset, f)

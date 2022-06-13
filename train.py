import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import numpy as np
import time
import argparse
import os
import random
import torch.backends.cudnn
from torchvision import transforms
from tqdm import tqdm
import timm
import matplotlib.pyplot as plt
import data.dataset
# from loss_functions import ArcMarginProduct, FocalLoss
from torchtoolbox.nn.metric_loss import ArcLoss
from torchtoolbox.nn.loss import FocalLoss

parser = argparse.ArgumentParser()

parser.add_argument('--batch-size', default=200, type=int)
parser.add_argument('--shot', default=1, type=int)
parser.add_argument('--scale', default=1, type=float)
parser.add_argument('--reprob', default=0.2, type=float)
parser.add_argument('--ra-m', default=12, type=int)
parser.add_argument('--ra-n', default=2, type=int)
parser.add_argument('--jitter', default=0.2, type=float)

parser.add_argument('--hdim', default=256, type=int)
parser.add_argument('--depth', default=16, type=int)
parser.add_argument('--psize', default=2, type=int)
parser.add_argument('--conv-ks', default=5, type=int)
parser.add_argument('--enlarge', default=20, type=int)

parser.add_argument('--wd', default=0.005, type=float)
parser.add_argument('--clip-norm', default=True, action='store_true')
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--lr-max', default=0.001, type=float)
parser.add_argument('--workers', default=2, type=int)

parser.add_argument('--device', default='0', type=str)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device
seed_value = 0
np.random.seed(seed_value)
torch.manual_seed(seed_value)
random.seed(seed_value)
if args.device != 'cpu':
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(244, scale=(args.scale, 1.0), ratio=(1.0, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandAugment(num_ops=args.ra_n, magnitude=args.ra_m),
    transforms.ColorJitter(args.jitter, args.jitter, args.jitter),
    torchvision.transforms.ToTensor(),
])

basic_transform = transforms.Compose([
    transforms.Resize(244),
    torchvision.transforms.ToTensor(),
])

test_transfrom = transforms.Compose([
    transforms.Resize((244, 244)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.84056849, 0.84056849, 0.84056849],
    #                      std=[0.30429188, 0.30429188, 0.30429188]),
])

print("Loading training data ...")
trainset = data.dataset.OracleFS(dataset_type='train', shot=args.shot, transform=train_transform,
                                 enlarge=args.enlarge, Normalize=False)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=args.workers)

print("Loading testing data ...")
testset = data.dataset.OracleFS(dataset_type='test', shot=args.shot, transform=test_transfrom, Normalize=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=args.workers)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Sequential(
            # nn.Linear(512, 64)
            # nn.Dropout(p=0.9),
        )
        self.loss = ArcLoss(512, 200)

    def forward(self, input, label):
        feature = self.model(input)
        similarity = self.loss.linear(feature)
        output = self.loss(feature, label)
        return output, similarity


if __name__ == '__main__':
    model = Net().cuda()
    opt = optim.Adam(model.parameters(), lr=args.lr_max)
    criterion = FocalLoss(200, gamma=2)
    scaler = torch.cuda.amp.GradScaler()
    lr_schedule = lambda t: np.interp([t], [0, args.epochs * 2 // 5, args.epochs * 4 // 5, args.epochs],
                                      [0, args.lr_max, args.lr_max / 20.0, 0])[0]

    train_acc_list = []
    test_acc_list = []
    loss_list = []
    lr_list = []
    for epoch in range(args.epochs):
        start = time.time()
        train_loss, train_acc, n, loss_value, lr = 0, 0, 0, 0, 0
        for i, (X, y) in enumerate(tqdm(trainloader, ncols=0)):
            model.train()
            X, y = X.cuda(), y.cuda()

            lr = lr_schedule(epoch + (i + 1) / len(trainloader))
            opt.param_groups[0].update(lr=lr)

            opt.zero_grad()
            output, similarity = model(X, y)
            loss = criterion(output, y)

            scaler.scale(loss).backward()
            if args.clip_norm:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(opt)
            scaler.update()

            train_loss += loss.item() * y.size(0)
            train_acc += torch.eq(similarity.max(1)[1], y).sum().item()
            loss_value += loss.item()
            n += y.size(0)

        model.eval()
        test_acc, m = 0, 0

        with torch.no_grad():
            for i, (X, y) in enumerate(testloader):
                X, y = X.cuda(), y.cuda()
                output, similarity = model(X, y)
                test_acc += torch.eq(similarity.max(1)[1], y).sum().item()
                m += y.size(0)

        train_acc_list.append(train_acc / n)
        loss_list.append(loss_value / n)
        test_acc_list.append(test_acc / m)
        lr_list.append(lr)

        print(
            f'[{args.shot} Shot] Epoch: {epoch} | Train Acc: {train_acc / n:.4f},'
            f'Test Acc: {test_acc / m:.4f}, loss: {loss_value / n:.4f}, '
            f'Time: {time.time() - start:.1f}, lr: {lr:.6f}')

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].plot(range(args.epochs), train_acc_list, label='train acc')
    axes[0].plot(range(args.epochs), test_acc_list, label='test acc')
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('accuracy')
    axes[0].set_title('{shot} Shot Accuracy'.format(shot=args.shot))

    axes[1].plot(range(args.epochs), loss_list, label='train loss')
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('loss')
    axes[1].set_title('{shot} Shot Loss'.format(shot=args.shot))

    fig.legend()
    fig.savefig('output/{shot}_shot.jpg'.format(shot=args.shot), dpi=500)

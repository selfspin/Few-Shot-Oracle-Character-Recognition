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
from tqdm import tqdm
import matplotlib.pyplot as plt
import data.dataset

parser = argparse.ArgumentParser()

parser.add_argument('--batch-size', default=10, type=int)
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

parser.add_argument('--wd', default=0.005, type=float)
parser.add_argument('--clip-norm', action='store_true')
parser.add_argument('--epochs', default=20, type=int)
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

trainset = data.dataset.OracleFS(dataset_type='train', shot=args.shot)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=args.workers)

testset = data.dataset.OracleFS(dataset_type='test', shot=args.shot)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=args.workers)

model = torchvision.models.resnet18(pretrained=True)
num_feature = model.fc.in_features
model.fc = torch.nn.Linear(num_feature, 200)
model.cuda()

# lr_schedule = lambda t: np.interp([t], [0, args.epochs * 2 // 5, args.epochs * 4 // 5, args.epochs],
#                                   [0, args.lr_max, args.lr_max / 20.0, 0])[0]

opt = optim.Adam(model.parameters(), lr=args.lr_max)
criterion = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler()

train_acc_list = []
test_acc_list = []
lr_list = []
for epoch in range(args.epochs):
    start = time.time()
    train_loss, train_acc, n = 0, 0, 0
    for i, (X, y) in enumerate(trainloader):
        model.train()
        X = F.interpolate(X, size=(224, 224), mode='bilinear')
        X, y = X.cuda(), y.cuda()

        # lr = lr_schedule(epoch + (i + 1) / len(trainloader))
        # opt.param_groups[0].update(lr=lr)

        lr = opt.state_dict()['param_groups'][0]['lr']

        opt.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(X)
            loss = criterion(output, y)

        scaler.scale(loss).backward()
        if args.clip_norm:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        train_loss += loss.item() * y.size(0)
        train_acc += (output.max(1)[1] == y).sum().item()
        n += y.size(0)

    model.eval()
    test_acc, m = 0, 0
    with torch.no_grad():
        for i, (X, y) in enumerate(testloader):
            X = F.interpolate(X, size=(224, 224), mode='bilinear')
            X, y = X.cuda(), y.cuda()
            with torch.cuda.amp.autocast():
                output = model(X)
            test_acc += (output.max(1)[1] == y).sum().item()
            m += y.size(0)

    train_acc_list.append(train_acc / n)
    test_acc_list.append(test_acc / m)
    lr_list.append(lr)

    print(
        f'[{args.shot} Shot] Epoch: {epoch} | Train Acc: {train_acc / n:.4f}, '
        f'Test Acc: {test_acc / m:.4f}, Time: {time.time() - start:.1f}, lr: {lr:.6f}')

plt.figure()
plt.plot(range(args.epochs), train_acc_list, label='train acc')
plt.plot(range(args.epochs), test_acc_list, label='test acc')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('{shot} Shot Accuracy'.format(shot=args.shot))
plt.legend()

plt.savefig('output/{shot}_shot.jpg'.format(shot=args.shot), dpi=500)

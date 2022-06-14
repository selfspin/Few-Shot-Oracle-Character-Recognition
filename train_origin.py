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
import data.dataset_masked
from torch.autograd import Variable
import pickle
import data.loading

# from torchtoolbox.transform import Cutout
# from torchtoolbox.tools import cutmix_data, mixup_criterion

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
    # transforms.RandomVerticalFlip(p=0.5),
    transforms.RandAugment(num_ops=args.ra_n, magnitude=args.ra_m),
    transforms.ColorJitter(args.jitter, args.jitter, args.jitter),
    # Cutout(),
    torchvision.transforms.ToTensor(),
])

test_transfrom = transforms.Compose([
    transforms.Resize((244, 244)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.84056849, 0.84056849, 0.84056849],
    #                      std=[0.30429188, 0.30429188, 0.30429188]),
])

print("Loading training data ...")
# trainset = data.dataset_masked.OracleFS_Masked(shot=args.shot, transform=train_transform,
#                                                enlarge=args.enlarge, Normalize=False)
trainset = data.dataset.OracleFS(dataset_type='train', shot=args.shot, transform=train_transform,
                                 enlarge=args.enlarge, Normalize=False)
# with open('data/1_shot_train.pickle', 'rb') as f:
#     trainset = pickle.load(f)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=args.workers)

print("Loading testing data ...")
testset = data.dataset.OracleFS(dataset_type='test', shot=args.shot, transform=test_transfrom, Normalize=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=args.workers)

# # RESNET
# class similarity(nn.Module):
#     def __init__(self, in_features, out_features):
#         super(similarity, self).__init__()
#         self.weight = torch.nn.Parameter(torch.FloatTensor(out_features, in_features))
#         nn.init.xavier_uniform_(self.weight)
#
#     def forward(self, input):
#         # --------------------------- cos(theta) & phi(theta) ---------------------------
#         return F.linear(F.normalize(input), self.weight)
model = torchvision.models.resnet18(pretrained=True)
model.fc = nn.Sequential(
    nn.Dropout(p=0.9),
    nn.Linear(512, 200)
)


# efficient net
# model = torchvision.models.efficientnet_b0(pretrained=True)
# model.classifier = nn.Sequential(
#     nn.Dropout(p=0.2),
#     # l2_norm(),
#     torch.nn.Linear(1280, 200)
# )
# criterion = nn.CrossEntropyLoss()


# VGG net
# model = torchvision.models.vgg11_bn(pretrained=True)
# num_feature = model.classifier[6].in_features
# model.classifier[6] = torch.nn.Linear(num_feature, 200)

# model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=200)
# for name, param in model.named_parameters():
#     if 'head' not in name:
#         param.requires_grad = False


# opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_max)
# opt = optim.SGD(model.parameters(), lr=args.lr_max, momentum=0.9, nesterov=True, weight_decay=1e-3)
# opt = optim.Adam(model.parameters(), lr=args.lr_max, weight_decay=1e-3)

# def criterion(outputs, label):
#     alpha = 0.0001
#     loss_fun = nn.CrossEntropyLoss()
#     loss = loss_fun(outputs, label)
#     for group in opt.param_groups:
#         for p in group['params']:
#             if p.grad is not None:
#                 loss += alpha * torch.abs(p).sum().item()
#     return loss


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """

    def __init__(self, class_num=200, alpha=Variable(torch.ones(200, 1) * 0.25), gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        probs = (P * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()
        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        return batch_loss.sum()


if __name__ == '__main__':
    model.cuda()
    criterion = nn.CrossEntropyLoss()
    # criterion = FocalLoss()
    # criterion = LabelSmoothingLoss(200, 0.005)
    opt = optim.Adam(model.parameters(), lr=args.lr_max, weight_decay=0.001)
    # opt = optim.Adam(model.parameters(), lr=args.lr_max)
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
            # lr = opt.state_dict()['param_groups'][0]['lr']

            opt.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(X)
                loss = criterion(outputs, y)

            scaler.scale(loss).backward()
            if args.clip_norm:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(opt)
            scaler.update()

            train_loss += loss.item() * y.size(0)
            train_acc += torch.eq(outputs.max(1)[1], y).sum().item()
            loss_value += loss.item()
            n += y.size(0)

        model.eval()
        test_acc, m = 0, 0
        with torch.no_grad():
            for i, (X, y) in enumerate(testloader):
                X, y = X.cuda(), y.cuda()
                with torch.cuda.amp.autocast():
                    output = model(X)
                test_acc += torch.eq(output.max(1)[1], y).sum().item()
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

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm
import time
import numpy as np
import torchvision.transforms as T
from DNN_printer import DNN_printer
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--name', type=str, default="SelfSpin")

parser.add_argument('--batch-size', default=512, type=int)
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
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--lr-max', default=0.05, type=float)
parser.add_argument('--workers', default=2, type=int)

if __name__ == '__main__':

    torch.manual_seed(0)  # 为CPU设置随机种子
    torch.cuda.manual_seed_all(0)  # 为所有GPU设置随机种子

    args = parser.parse_args()
    device = torch.device('cuda')

    if args.name == 'ConvMixer':
        model = ConvMixer(args.hdim, args.depth, patch_size=args.psize, kernel_size=args.conv_ks, n_classes=10)
        model.load_state_dict(torch.load('ConvMixer_final.pth'))
    else:
        model = SelfSpin(num_classes=10)
        model.load_state_dict(torch.load('SelfSpin_final.pth'))
    model.to(device)

    model.eval()
    DNN_printer(model, (3, 32, 32), 1024)

    normalize = T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    cifar_test_transform = T.Compose([
        T.ToTensor(),
        normalize,
    ])
    cifar_data_path = './data'
    cifar_test = CIFAR10(cifar_data_path, train=False, transform=cifar_test_transform)
    cifar_test_loader = DataLoader(cifar_test, batch_size=1024, num_workers=2, pin_memory=False)

    test_acc, m = 0, 0
    with torch.no_grad():
        for i, (X, y) in enumerate(cifar_test_loader):
            X, y = X.cuda(), y.cuda()
            with torch.cuda.amp.autocast():
                output = model(X)
            test_acc += (output.max(1)[1] == y).sum().item()
            m += y.size(0)

    print('Final model\'s test acc: {:.4f}'.format(test_acc / m))

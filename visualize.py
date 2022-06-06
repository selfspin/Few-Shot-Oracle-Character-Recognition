import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--name', type=str, default="ConvMixer")

parser.add_argument('--batch-size', default=4, type=int)

parser.add_argument('--hdim', default=256, type=int)
parser.add_argument('--depth', default=16, type=int)
parser.add_argument('--psize', default=2, type=int)
parser.add_argument('--conv-ks', default=5, type=int)

if __name__ == '__main__':

    torch.manual_seed(0)  # 为CPU设置随机种子
    torch.cuda.manual_seed_all(0)  # 为所有GPU设置随机种子

    args = parser.parse_args()
    device = torch.device('cpu')

    if args.name == 'ConvMixer':
        model = ConvMixer(args.hdim, args.depth, patch_size=args.psize,
                          kernel_size=args.conv_ks, n_classes=10)
        model.load_state_dict(torch.load('ConvMixer_final.pth'))
    else:
        model = SelfSpin(num_classes=10)
        model.load_state_dict(torch.load('SelfSpin_final.pth'))
    model.to(device)

    model.eval()

    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    normalize = T.Normalize(mean=mean, std=std)
    cifar_test_transform = T.Compose([
        T.ToTensor(),
        normalize,
    ])
    cifar_data_path = './data'
    cifar_test = CIFAR10(cifar_data_path, train=False, transform=cifar_test_transform)
    cifar_test_loader = DataLoader(cifar_test, batch_size=4, num_workers=2, pin_memory=False)


    def imshow(img, mean, std, transpose=True):
        std = torch.tensor(std)[:, None, None]
        mean = torch.tensor(mean)[:, None, None]
        npimg = (img * std + mean).numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


    dataiter = iter(cifar_test_loader)
    images, labels = dataiter.next()

    # print images
    imshow(torchvision.utils.make_grid(images), mean, std)
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    outputs = model(images)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(4)))

    # -------------------------------------------------------

    ind = 1
    input = images[ind].unsqueeze(0)
    input.requires_grad = True
    model.eval()


    def attribute_image_features(algorithm, input, **kwargs):
        model.zero_grad()
        tensor_attributions = algorithm.attribute(input,
                                                  target=labels[ind],
                                                  **kwargs
                                                  )

        return tensor_attributions


    saliency = Saliency(model)
    grads = saliency.attribute(input, target=labels[ind].item())
    grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))

    ig = IntegratedGradients(model)
    attr_ig, delta = attribute_image_features(ig, input, baselines=input * 0,
                                              return_convergence_delta=True)
    attr_ig = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))
    print('Approximation delta: ', abs(delta))

    ig = IntegratedGradients(model)
    nt = NoiseTunnel(ig)
    attr_ig_nt = attribute_image_features(nt, input, baselines=input * 0, nt_type='smoothgrad_sq',
                                          nt_samples=100, stdevs=0.2)
    attr_ig_nt = np.transpose(attr_ig_nt.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

    dl = DeepLift(model)
    attr_dl = attribute_image_features(dl, input, baselines=input * 0)
    attr_dl = np.transpose(attr_dl.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

    print('Original Image')
    print('Predicted:', classes[predicted[ind]],
          ' Probability:', torch.max(torch.nn.functional.softmax(outputs, 1)).item())

    original_image = np.transpose((images[ind].cpu().detach().numpy()
                                   * np.array(std)[:, None, None])
                                  + np.array(mean)[:, None, None], (1, 2, 0))

    _ = viz.visualize_image_attr(None, original_image,
                                 method="original_image", title="Original Image")

    _ = viz.visualize_image_attr(grads, original_image, method="blended_heat_map", sign="absolute_value",
                                 show_colorbar=True, title="Overlayed Gradient Magnitudes")

    _ = viz.visualize_image_attr(attr_ig, original_image, method="blended_heat_map", sign="all",
                                 show_colorbar=True, title="Overlayed Integrated Gradients")

    _ = viz.visualize_image_attr(attr_ig_nt, original_image, method="blended_heat_map", sign="absolute_value",
                                 outlier_perc=10, show_colorbar=True,
                                 title="Overlayed Integrated Gradients \n with SmoothGrad Squared")

    _ = viz.visualize_image_attr(attr_dl, original_image, method="blended_heat_map", sign="all", show_colorbar=True,
                                 title="Overlayed DeepLift")
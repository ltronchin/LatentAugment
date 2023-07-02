from typing import Sequence

from itertools import chain

import torch
import torch.nn as nn
from torchvision import models

from augments.criteria.lpips.utils import normalize_activation
from utils import util_reports

def get_network(net_type: str, report_dir: str):
    if net_type == 'alex':
        return AlexNet()
    elif net_type == 'squeeze':
        return SqueezeNet()
    elif net_type == 'vgg':
        return VGG16(report_dir)
    else:
        raise NotImplementedError('choose net_type from [alex, squeeze, vgg].')

class LinLayers(nn.ModuleList):
    def __init__(self, n_channels_list: Sequence[int]):
        super(LinLayers, self).__init__([
            nn.Sequential(
                nn.Identity(),
                nn.Conv2d(nc, 1, 1, 1, 0, bias=False)
            ) for nc in n_channels_list
        ])

        for param in self.parameters():
            param.requires_grad = False


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()

        # register buffer
        self.register_buffer(
            'mean', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer(
            'std', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def set_requires_grad(self, state: bool):
        for param in chain(self.parameters(), self.buffers()):
            param.requires_grad = state

    def z_score(self, x: torch.Tensor):
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor):
        x = self.z_score(x)

        output = []
        for i, (_, layer) in enumerate(self.layers._modules.items(), 1):
            x = layer(x)
            #util_reports.show_activation(x[0], i, self.report_dir)
            if i in self.target_layers:
                output.append(normalize_activation(x))
            if len(output) == len(self.target_layers):
                break
        return output

class SqueezeNet(BaseNet):
    def __init__(self):
        super(SqueezeNet, self).__init__()

        self.layers = models.squeezenet1_1(True).features
        self.target_layers = [2, 5, 8, 10, 11, 12, 13]
        self.n_channels_list = [64, 128, 256, 384, 384, 512, 512]

        self.set_requires_grad(False)


class AlexNet(BaseNet):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.layers = models.alexnet(True).features
        self.target_layers = [2, 5, 8, 10, 12]
        self.n_channels_list = [64, 192, 384, 256, 256]

        self.set_requires_grad(False)


class VGG16(BaseNet):
    def __init__(self, report_dir):
        super(VGG16, self).__init__()
        
        self.report_dir = report_dir

        self.layers = models.vgg16(True).features
        self.target_layers =[16, 23, 30] # [16, 23, 30] # [4, 9, 16, 23, 30]
        self.n_channels_list = [256, 512, 512] #  # [64, 128, 256, 512, 512]

        self.set_requires_grad(False)
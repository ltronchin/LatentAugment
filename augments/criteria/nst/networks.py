import torch
import torch.nn as nn
import torchvision.models as models


class VGG19Net(nn.Module):
    def __init__(self, device):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(VGG19Net, self).__init__()
        self.cnn = models.vgg19(pretrained=True).features.eval().requires_grad_(False).to(device)
        self.content_layers = ['19'] # ['conv_9']
        self.style_layers = ['0', '5', '10', '19', '28'] # ['conv_1', 'conv_3', 'conv_5', 'conv_9', 'conv_13']
        self.mean= torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(device)
        self.std= torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(device)

    def print_layer(self):
        i = 0
        name = []
        for layer in self.cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name.append('conv_{}'.format(i))
            elif isinstance(layer, nn.ReLU):
                name.append('relu_{}'.format(i))
            elif isinstance(layer, nn.MaxPool2d):
                name.append('pool_{}'.format(i))
            elif isinstance(layer, nn.BatchNorm2d):
                name.append('bn_{}'.format(i))
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
        print(name)
        return name

    def prepare_img(self, img, mode_idx, synthetic, normalize, standardize=False):

        # Rescale in range [0 1]
        if synthetic:
            img = (img * 127.5 + 128).clamp(0, 255)
        if normalize:
            img = img / 255.0
        assert img.min().item() >= 0.0
        assert img.max().item() <= 1.0

        # Select mode
        if mode_idx is not None:
            x = img[:, mode_idx, :, :].unsqueeze(dim=1)
        else:
            x = img

        # Make tree channel.
        if x.shape[1] == 1:
            x = x.repeat([1, 3, 1, 1])  # make a three channel tensor

        # Normalize
        if standardize:
            x = (x - self.mean) / self.std  # Normalize img

        return x

    def forward(self, x, mode_dict=None, synthetic=False, normalize=False):
        """Extract multiple convolutional feature maps."""
        x = self.prepare_img(x, mode_idx=mode_dict['mode_idx'], synthetic=synthetic, normalize=normalize)
        f_style, f_content = [], []
        for name, layer in self.cnn._modules.items():
            x = layer(x)
            if name in self.style_layers:
                f_style.append(x)
            if name in self.content_layers:
                f_content.append(x)
        return f_style, f_content
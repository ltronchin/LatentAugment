import torch
import torch.nn as nn

from augments.criteria.lpips.networks import get_network, LinLayers
from augments.criteria.lpips.utils import get_state_dict_custom

def l2_loss_vectorized(X, Y, compute_mean=True):

    assert X.ndim == 4
    assert Y.ndim == 4
    m, c, h, w = Y.shape
    n, _, _, _ = X.shape
    YYt = torch.sum(torch.square(Y), (1, 2, 3))
    XXt = torch.sum(torch.square(X), (1, 2, 3))
    YXt = torch.einsum('nchw, mchw -> nm', [Y, X])

    D = (YYt.unsqueeze(dim=-1) + XXt) - (2 * YXt)

    if compute_mean:
        D = torch.sum(D) / (m * n)  # L2 distance
        D = D / (c * h * w)  # normalization factor

    return D

class LPIPS(nn.Module):
    r"""Creates a criterion that measures
    Learned Perceptual Image Patch Similarity (LPIPS).
    Arguments:
        net_type (str): the network type to compare the features:
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """
    def __init__(self, net_type: str = 'alex', report_dir:str = None, version: str = '0.1'):

        assert version in ['0.1'], 'v0.1 is only supported now'

        super(LPIPS, self).__init__()

        # pretrained network
        self.net = get_network(net_type, report_dir).to("cuda")

        # linear layers
        self.lin = LinLayers(self.net.n_channels_list).to("cuda")

        self.lin.load_state_dict(get_state_dict_custom(net_type, version, nlin=len(self.lin)))

        self.target_layers = self.net.target_layers

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        feat_x, feat_y = self.net(x), self.net(y)

        diff = [
            (fx - fy) ** 2 for fx, fy in zip(feat_x, feat_y)
        ]

        res = [l(d).mean((2, 3), True) for d, l in zip(diff, self.lin)]

        return torch.sum(torch.cat(res, 0)) / max(x.shape[0], y.shape[0])

    def forward_tr(self, x: torch.Tensor, feat: list):
        feat_x = self.net(x)

        diff = [
            (fx - fy) ** 2 for fx, fy in zip(feat_x, feat)
        ]
        res = [l(d).mean((2, 3), True) for d, l in zip(diff, self.lin)]

        return torch.sum(torch.cat(res, 0)) / feat[0].shape[0]

    def extract_features_shape(self, x: torch.Tensor, lin: int):
        feat = self.net(x)
        return feat[lin].shape

    def extract_features(self, x: torch.Tensor):
        feat = self.net(x)
        return feat

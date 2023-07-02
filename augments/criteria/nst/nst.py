import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
def gram_matrix(f):
    _, c, h, w  = f.size()  # a: batch size(=1), b: number of feature maps (c,d): dimensions of a features map (N = c * d)

    features = f.view(c, h * w)  # resise F_XL into \hat F_XL
    G = torch.mm(features, features.t())  # compute the gram product
    #return G / (c * h * w) # we 'normalize' the values of the gram matrix by dividing by the number of element in each feature maps.
    return G / (h * w)
    #return G
# ------------------------------------------------------------

class NSTLoss(nn.Module):
    def __init__(self, batch_size):
        super(NSTLoss, self).__init__()
        self.style_loss = 0.0
        self.content_loss = 0.0
        self.batch_size = batch_size

    def zero_loss(self):
        self.style_loss = 0.0
        self.content_loss = 0.0

    def _style_loss(self, f1, f2):
        G1 = gram_matrix(f1)
        G2 = gram_matrix(f2)

        self.style_loss += F.mse_loss(G1, G2)

    def _content_loss(self, f1, f2):
        self.content_loss += F.mse_loss(f1, f2)

    def info(self):
        print()
        print("Style loss: {:4f}".format(self.style_loss.item()))
        print("Content loss: {:4f}".format(self.content_loss.item()))
        print()

    def forward(self, f_style, f_content, f_style_target, f_content_target, verbose=False):
        assert len(f_content) == 1
        assert len(f_content_target) == 1

        for idx_batch in range(self.batch_size): # vectorize the operation
            xc  = f_content[0][idx_batch, :, :, :].unsqueeze(0)
            xct = f_content_target[0][idx_batch, :, :, :].unsqueeze(0)
            self._content_loss(xc, xct)
            for fs, fst in zip(f_style, f_style_target):
                xs  = fs[idx_batch, :, :, :].unsqueeze(0)
                xst = fst[idx_batch, :, :, :].unsqueeze(0)
                self._style_loss(xs, xst)

        self.content_loss /= self.batch_size
        self.style_loss /= self.batch_size

        if verbose:
            self.info()






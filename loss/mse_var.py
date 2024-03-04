import torch
from torch import nn


class MSE_VAR(nn.Module):
    """Loss function  according to the paper
    Kendall, A., & Gal, Y. (2017). What uncertainties do we need in
    bayesian deep learning for computer vision?. Advances in neural
    information processing systems, 30."""
    def __init__(self):
        super(MSE_VAR, self).__init__()

    def forward(self, mean, var, inputs):
        loss1 = torch.mul(torch.exp(-var), torch.pow(mean - inputs, 2))
        loss2 = var
        loss = 0.5 * (loss1 + loss2)

        return loss

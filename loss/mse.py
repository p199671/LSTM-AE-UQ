from torch import nn


class MSE(nn.Module):
    """Mean Squared Error loss function"""
    def __init__(self):
        super(MSE, self).__init__()
        self.loss = nn.MSELoss(reduction='none')

    def forward(self, reconstructions, inputs):
        return self.loss(reconstructions, inputs)

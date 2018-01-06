from torch import nn
import torch.nn.functional as F


# loss function: seven probability map --- 6 scale + 1 fuse
class Loss(nn.Module):
    def __init__(self, weight=[1.0] * 7):
        super(Loss, self).__init__()
        self.weight = weight

    def forward(self, x_list, label):
        loss = self.weight[0] * F.binary_cross_entropy(x_list[0], label)
        for i, x in enumerate(x_list[1:]):
            loss += self.weight[i + 1] * F.binary_cross_entropy(x, label)
        return loss

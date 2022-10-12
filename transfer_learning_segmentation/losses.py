# PyTorch
from torch import nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    def __init__(self, weight=1, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.weight = weight

    def forward(self, inputs, targets, smooth=1):
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = self.weight*dice_loss + (1-self.weight)*BCE

        return Dice_BCE
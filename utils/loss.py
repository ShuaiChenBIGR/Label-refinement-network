import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np


class DiceCoefficientLF(nn.Module):

    def __init__(self):
        super(DiceCoefficientLF, self).__init__()

    def forward(self, y_true, y_pred):
        return 1.0 - DiceCoeff().forward(y_true, y_pred)


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        # input = input * weighting
        self.inter = torch.dot(input.reshape(-1), target.reshape(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def Dice_numpy(inputs, labels):
    eps = 0.0001
    inter = np.dot(inputs.flatten(), labels.flatten())
    union = np.sum(inputs) + np.sum(labels) + eps

    dice = (2 * inter + eps) / union

    return dice

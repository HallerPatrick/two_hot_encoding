import torch
from torch import nn


class CrossEntropyLossSoft(nn.Module):

    def __init__(self, reduction="mean", ignore_index=None):
        super(CrossEntropyLossSoft, self).__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, input, target):
        """
        Args:
            input: (batch, *)
            target: (batch, *) same shape as input, each item must be a valid distribution: target[i, :].sum() == 1.

        """
        if self.ignore_index:
            target[:, self.ignore_index] = 0

        logprobs = torch.nn.functional.log_softmax(input.view(input.shape[0], -1), dim=1)
        batchloss = - torch.sum(target.view(target.shape[0], -1) * logprobs, dim=1)
        if self.reduction == 'none':
            return batchloss
        elif self.reduction == 'mean':
            return torch.mean(batchloss)
        elif self.reduction == 'sum':
            return torch.sum(batchloss)
        else:
            raise NotImplementedError('Unsupported reduction mode.')



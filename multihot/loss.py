import torch
from torch import nn


class CrossEntropyLossSoft(nn.Module):
    def __init__(self, ignore_index=None, weight=None):
        super(CrossEntropyLossSoft, self).__init__()
        self.ignore_index = ignore_index
        self.weight = weight

    def forward(self, input, target):
        """
        Args:
            input: (batch, *)
            target: (batch, *) same shape as input, each item must be a valid distribution: target[i, :].sum() == 1.

        """
        if self.ignore_index:
            target[:, self.ignore_index] = 0

        logprobs = nn.functional.log_softmax(input.view(input.shape[0], -1), dim=1)

        # Calculate logprobs for each class with weights
        if self.weight is not None:
            logprobs = logprobs * self.weight

        # Calculate loss
        batchloss = -torch.sum(target.view(target.shape[0], -1) * logprobs, dim=1)

        return torch.mean(batchloss)

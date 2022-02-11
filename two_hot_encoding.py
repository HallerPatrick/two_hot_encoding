import torch

import torch.nn.functional as F
from torch import nn


def two_hot(input_one, input_two, num_clases):
    assert input_one.device == input_two.device
    shape = list(input_one.size())

    shape.append(num_clases)
    ret = torch.zeros(shape).to(input_one.device)

    # Unigrams
    ret.scatter_(-1, input_one.unsqueeze(-1), 1)
    # Bigrams
    ret.scatter_(-1, input_two.unsqueeze(-1), 1)
    # ret.scatter_(-1, input_two[input_two != -1], 1)
    # input_two = input_two.unsqueeze(-1)

    return ret


def n_hot(t, num_clases):
    shape = list(t.size())[1:]

    shape.append(num_clases)
    ret = torch.zeros(shape).to(t.device)

    # Expect that first dimension is for all n-grams
    for seq in t:
        ret.scatter_(-1, seq.unsqueeze(-1), 1)

    return ret


def soft_n_hot(input, num_classes):
    soft_dist = 1 / input.size(0)
    shape = list(input.size())[1:]

    shape.append(num_classes)
    ret = torch.zeros(shape).to(input.device)

    for t in input:
        ret.scatter_(-1, t.unsqueeze(-1), soft_dist)

    return ret


def soft_two_hot(input_one, input_two, num_classes):
    assert input_one.device == input_two.device
    shape = list(input_one.size())

    shape.append(num_classes)
    ret = torch.zeros(shape).to(input_one.device)

    # Unigrams
    ret.scatter_(-1, input_one.unsqueeze(-1), 0.5)

    # Bigrams
    ret.scatter_(-1, input_two.unsqueeze(-1), 0.5)

    return ret


class NGramsEmbedding(nn.Module):
    """N-Hot encoder"""

    def __init__(self, num_embeddings: int, embedding_dim):
        super(NGramsEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.num_classes = num_embeddings

    @property
    def weight(self):
        return self.embedding.weight

    def forward(self, input: torch.Tensor, **kwargs):
        # TODO: linear bias?
        return self._forward(n_hot(input, self.num_classes, **kwargs))

    def _forward(self, n_hot: torch.Tensor) -> torch.Tensor:
        return F.linear(n_hot, self.embedding.weight.t())


class TwoHotEmbedding(nn.Module):
    """Two hot encoder of unigrams and bigrams"""

    def __init__(self, num_embeddings: int, embedding_dim):
        super(TwoHotEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.num_classes = num_embeddings

    @property
    def weight(self):
        return self.embedding.weight

    def forward(self, input_one: torch.Tensor, input_two: torch.Tensor, **kwargs):

        # TODO: linear bias?
        return self._forward(two_hot(input_one, input_two, self.num_classes, **kwargs))

    def _forward(self, two_hot: torch.Tensor) -> torch.Tensor:
        return F.linear(two_hot, self.embedding.weight.t())

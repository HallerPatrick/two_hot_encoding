from functools import lru_cache

import torch

import torch.nn.functional as F
from torch import nn


try:
    import ngram

    n_hot = ngram.n_hot
    soft_n_hot = ngram.n_soft_hot

except ImportError:
    def n_hot(t, num_clases):
        shape = list(t.size())[1:]

        shape.append(num_clases)
        ret = torch.zeros(shape).to(t.device)

        # Expect that first dimension is for all n-grams
        for seq in t:
            ret.scatter_(-1, seq.unsqueeze(-1), 1)

        return ret

    def soft_n_hot(input: torch.Tensor, num_classes, soft_labels: torch.Tensor):
        # soft_dist = 1 / input.size(0)

        shape = list(input.size())[1:]

        shape.append(num_classes)

        ret = torch.zeros(shape).to(input.device)


        for i, t in enumerate(input):
            ret.scatter_(-1, t.unsqueeze(-1), soft_labels[i].item())

        return ret


@lru_cache(maxsize=5)
def soft_dist(n, device):
    return torch.tensor([1 / n] * n).to(device)


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



import torch

from torch import nn
import torch.nn.functional as F

def two_hot(input_one, input_two, num_clases):
    shape = list(input_one.size())

    shape.append(num_clases)
    ret = torch.zeros(shape)

    ret.scatter_(-1, input_one.unsqueeze(-1), 1)
    ret.scatter_(-1, input_two.unsqueeze(-1), 1)

    return ret


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
        """
        Thats the plan:
            1. Pass input through get the encoded idx
            2. Calculate the binary index and extract to unigram and bigram indexes
            3. Construct a two hot encoding in one vector
            4. Pass through a embedding layer
        """

        # TODO: linear bias?
        return self._forward(two_hot(input_one, input_two, self.num_classes, **kwargs))

    def _forward(self, two_hot: torch.Tensor) -> torch.Tensor:
        return F.linear(two_hot, self.embedding.weight.t())

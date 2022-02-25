import torch
import ngram




x = torch.zeros((1, 2, 2), dtype=torch.int64)

l = torch.tensor([1])

ngram.n_hot(x, 2)
ngram.n_soft_hot(x, 2, l)

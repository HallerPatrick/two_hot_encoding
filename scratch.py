import torch
import torch.nn.functional as F

from two_hot_encoding import target_dist


t = torch.tensor([-1., -1., 1., 1.])

# def target_distribution(t: torch.Tensor, dim=0) -> torch.Tensor:
#     bin_values = torch.count_nonzero(t)
#     
#     dist = 1 / bin_values
#
#     print(bin_values)
#

out = target_dist(torch.tensor([1]),torch.tensor([2]), 3)
print(out)

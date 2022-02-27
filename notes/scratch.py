import sys
import os
import torch
from torch import nn

sys.path.append(os.getcwd())

from two_hot_encoding import GramsEmbedding, NGramsEmbedding




def main():
    emb1 = GramsEmbedding(2, 8)

    emb2 = NGramsEmbedding(2, 8)
    
    t = torch.zeros((1, 2, 2), dtype=torch.int64)

    print(emb1(t).size())
    print(emb1(t))
    print(emb2(t).size())
    print(emb2(t))
    
    emb1.eval()
    emb2.eval()
    t = torch.zeros((1, 2, 2), dtype=torch.int64)

    print(emb1(t).size())
    print(emb1(t))
    print(emb2(t).size())
    print(emb2(t))





if __name__ == "__main__":
    main()

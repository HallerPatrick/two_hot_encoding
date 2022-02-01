import torch
from torch import nn


def main():
    t = torch.zeros((2, 3, 4))

    t[:, :, 2] = 99
    print(t)


if __name__ == "__main__":
    main()
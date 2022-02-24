import torch
from torch import nn

from data import Corpus

from datasets import load_dataset


def main():

    dataset = load_dataset("wikitext", "wikitext-103-v1")
    print(dataset)


if __name__ == "__main__":
    main()

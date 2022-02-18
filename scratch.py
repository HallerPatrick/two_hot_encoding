import torch
from torch import nn

from data import Corpus


def main():
    corpus = Corpus("data/love_song", ngrams=1)

    o = corpus.tokenize(["he"])

    print(corpus.dictionary.idx2word[o[0][0]])
    print(corpus.dictionary.idx2word[o[0][1]])


if __name__ == "__main__":
    main()

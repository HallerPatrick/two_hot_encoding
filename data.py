import os

from collections import Counter
from typing import List

import torch
from nltk import ngrams


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self._marker_tokens = []

    def add_word(self, word):
        if word.startswith("<") and word.endswith(">"):
            if word not in self._marker_tokens:
                self._marker_tokens.append(word)

        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def get_marker_tokens(self) -> List[str]:
        return self._marker_tokens

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, only_unigrams=False, unk_threshold=3):

        self.only_unigrams = only_unigrams
        self.unk_threshold = unk_threshold

        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, "train.txt"))
        self.valid = self.tokenize(os.path.join(path, "valid.txt"))
        self.test = self.tokenize(os.path.join(path, "test.txt"))

        if not self.only_unigrams:
            self.train_bigrams = self.tokenize_bigrams(os.path.join(path, "train.txt"))
            # self.display_text(self.train_bigrams)
            self.valid_bigrams = self.tokenize_bigrams(os.path.join(path, "valid.txt"))
            self.test_bigrams = self.tokenize_bigrams(os.path.join(path, "test.txt"))

    def display_text(self, t):
        for a in t:
            print(repr(self.dictionary.idx2word[a.item()]), end="")
        print()

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                words = list(line) + ["<eos>"]
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, "r", encoding="utf8") as f:
            idss = []
            for line in f:
                words = list(line) + ["<eos>"]
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids

    def setup_dictionary(self, path):
        assert os.path.exists(path)

        token_frequency = Counter()

        with open(path, "r") as f:
            lines = f.readlines()

        for line in lines:
            chars = list(line)

            for bigram in ngrams(chars, 2):
                c1, c2 = bigram
                token_frequency[c1] = -1
                token_frequency[c2] = -1

                # For now only keep track of bigram frequencies
                token_frequency[c1 + c2] += 1

        self.dictionary.add_word("<start>")
        self.dictionary.add_word("<eos>")
        self.dictionary.add_word("<UNK>")
        self.dictionary.add_word("<BI-UNK>")

        for toke, freq in token_frequency.items():
            if freq > self.unk_threshold or freq == -1:
                self.dictionary.add_word(toke)

    def tokenize_bigrams(self, path):
        """Tokenizes a text file."""

        # Add words to the dictionary
        self.setup_dictionary(path)

        # Tokenize file content
        with open(path, "r", encoding="utf8") as f:
            idss = []
            for line in f:
                words = ["<start>"] + list(line) + ["<eos>"]
                ids = []

                for i, word in enumerate(ngrams(words, 2)):
                    try:
                        ids.append(self.dictionary.word2idx[word[0] + word[1]])
                    except KeyError:
                        ids.append(self.dictionary.word2idx["<BI-UNK>"])

                idss.append(torch.tensor(ids).type(torch.int64))

            ids = torch.cat(idss)

        return ids


def grouped(iterable, n):
    # s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ...
    return zip(*[iter(iterable)] * n)

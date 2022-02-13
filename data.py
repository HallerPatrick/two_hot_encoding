import os
import sys

from collections import Counter, defaultdict
from pprint import pprint
from typing import List

import torch
from nltk import ngrams


class Dictionary:
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


class Corpus:
    def __init__(self, path, device="cpu", ngrams=2, unk_threshold=3):

        self.unk_threshold = unk_threshold
        self.ngrams = ngrams
        self.device = device

        # Keep track of all indexes for each ngram, this is used
        # for the generating task
        self.ngram_indexes = defaultdict(list)

        self.dictionary = Dictionary()

        self.train = self.tokenize(os.path.join(path, "train.txt"))
        self.valid = self.tokenize(os.path.join(path, "valid.txt"))
        self.test = self.tokenize(os.path.join(path, "test.txt"))

    def display_text(self, t):
        for a in t:
            print(repr(self.dictionary.idx2word[a.item()]), end="")
        print()

    def remove_marker_tokens(self, token):
        """Due to some str length comparison to determine what n-gram the token
        is. We replace the marker tokens, with a single char, for easy comparison
        """
        for marker in self.dictionary.get_marker_tokens():
            token = token.replace(marker, "i")

        return token

    def setup_dictionary(self, path):
        assert os.path.exists(path)

        token_frequency = Counter()

        with open(path, "r") as f:
            lines = f.readlines()

        for line in lines:
            chars = ["<start>"] + list(line) + ["<eos>"]
            for i in range(1, self.ngrams + 1):
                # Add UNK token for ngram
                self.dictionary.add_word(f"<{i}-UNK>")

                for ngram in ngrams(chars, i):
                    # Add all characters to frequencies dict
                    for c in ngram:
                        token_frequency[c] = -1

                    if len(ngram) != 1:
                        # For now only keep track of ngram frequencies
                        token_frequency["".join(ngram)] += 1

        self.dictionary.add_word("<start>")
        self.dictionary.add_word("<eos>")

        for toke, freq in token_frequency.items():
            if freq > self.unk_threshold or freq == -1:
                sanit_token = self.remove_marker_tokens(toke)
                idx = self.dictionary.add_word(toke)
                if idx not in self.ngram_indexes[len(sanit_token)]:
                    self.ngram_indexes[len(sanit_token)].append(idx)

    def tokenize(self, path):
        """Tokenizes a text file."""

        # Add words to the dictionary
        self.setup_dictionary(path)

        # Tokenize file content
        with open(path, "r", encoding="utf8") as f:
            lines = f.readlines()

        n_gram_sequences = []
        min_length = sys.maxsize

        for n in range(1, self.ngrams + 1):
            idss_n = []
            for line in lines:
                if n == 1:
                    words = list(line) + ["<eos>"]
                else:
                    # Adding start offsets for all ngrams
                    words = ["<start>" for _ in range(1, n)]
                    words.extend(list(line))
                    words.append("<eos>")

                ids = []
                length = 0
                for i, word in enumerate(ngrams(words, n)):
                    try:
                        ids.append(self.dictionary.word2idx["".join(word)])
                    except KeyError:
                        ids.append(self.dictionary.word2idx[f"<{n}-UNK>"])
                    length += 1

                idss_n.append(torch.tensor(ids).type(torch.int64))

            # N-gram sequence
            seq = torch.cat(idss_n).unsqueeze(dim=0)
            length = seq.size(1)

            if length < min_length:
                min_length = length

            n_gram_sequences.append(seq)

        n_gram_sequences = torch.cat([t[:min_length] for t in n_gram_sequences]).to(
            self.device
        )

        return n_gram_sequences


def grouped(iterable, n):
    # s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ...
    return zip(*[iter(iterable)] * n)

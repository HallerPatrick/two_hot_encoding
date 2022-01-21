import os
from io import open
import torch

from nltk import ngrams

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, only_unigrams=False):

        self.only_unigrams = only_unigrams

        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

        if not self.only_unigrams:
            self.train_bigrams = self.tokenize_bigrams(os.path.join(path, 'train.txt'))
            # self.display_text(self.train_bigrams)
            self.valid_bigrams = self.tokenize_bigrams(os.path.join(path, 'valid.txt'))
            self.test_bigrams = self.tokenize_bigrams(os.path.join(path, 'test.txt'))
    
    def display_text(self, t):
        for a in t:
            print(self.dictionary.idx2word[a.item()], end="")
        print()
            
    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = list(line) + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = list(line) + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids

    def tokenize_bigrams(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = list(line) + ['<eos>']
                for word in ngrams(words, 2):
                    self.dictionary.add_word(word[0] + word[1])

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = list(line) + ['<eos>']
                ids = []
                for word in ngrams(words, 2):
                    ids.append(self.dictionary.word2idx[word[0] + word[1]])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids


def grouped(iterable, n):
    # s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ...
    return zip(*[iter(iterable)] * n)

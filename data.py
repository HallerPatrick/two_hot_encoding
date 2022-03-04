import os
import sys

from collections import Counter, defaultdict
from operator import itemgetter
from pathlib import Path
from typing import List, Tuple

import torch
from nltk import ngrams
from tqdm import tqdm


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

    def add_item(self, word):
        return self.add_word(word)

    def get_marker_tokens(self) -> List[str]:
        return self._marker_tokens

    def __len__(self):
        return len(self.idx2word)

    def get_idx_for_item(self, item: str) -> int:
        """
        returns the ID of the string, otherwise 0
        :param item: string for which ID is requested
        :return: ID of string, otherwise 0
        """
        item = item.encode("utf-8")
        if item in self.word2idx.keys():
            return self.word2idx[item]
        else:
            return 0

    def get_idx_for_items(self, items: List[str]) -> List[int]:
        """
        returns the IDs for each item of the list of string, otherwise 0 if not found
        :param items: List of string for which IDs are requested
        :return: List of ID of strings
        """
        if not hasattr(self, "item2idx_not_encoded"):
            d = dict(
                [(key, value) for key, value in self.word2idx.items()]
            )
            self.item2idx_not_encoded = defaultdict(int, d)

        if not items:
            return []
        results = itemgetter(*items)(self.item2idx_not_encoded)
        if isinstance(results, int):
            return [results]
        return list(results)

    def get_items(self) -> List[str]:
        items = []
        for item in self.idx2word:
            items.append(item)
        return items

    def get_item_for_index(self, idx):
        return self.idx2word[idx]

    def save(self, savefile):
        import pickle

        with open(savefile, "wb") as f:
            mappings = {"idx2item": self.idx2word, "item2idx": self.word2idx}
            pickle.dump(mappings, f)

class Corpus:
    def __init__(self, path, device="cpu", ngrams=2, unk_threshold=3):

        self.unk_threshold = unk_threshold
        self.ngrams = ngrams
        self.device = device

        # Keep track of all indexes for each ngram, this is used
        # for the generating task
        self.ngram_indexes = defaultdict(list)

        self.dictionary = Dictionary()

        self.load_dataset(path)


    def load_dataset(self, path: str):

        if Path(path).is_dir():
            self.train = self.tokenize_file(os.path.join(path, "train.txt"))
            self.valid = self.tokenize_file(os.path.join(path, "valid.txt"))
            self.test = self.tokenize_file(os.path.join(path, "test.txt"))
        elif "data/enwik8" in str(path):
            prep_enwiki8()
            self.train = self.tokenize_file(os.path.join(path, "train.txt"))
            self.valid = self.tokenize_file(os.path.join(path, "valid.txt"))
            self.test = self.tokenize_file(os.path.join(path, "test.txt"))

        else:
            self.load_from_huggingface(str(path))



    def load_from_huggingface(self, path):
            from datasets import load_dataset

            name = path.split("/")

            # Load dataset
            dataset = load_dataset(*name)
            train = dataset["train"]["text"]
            valid = dataset["validation"]["text"]
            test = dataset["test"]["text"]
            
            sets = [("train", train), ("valid", valid), ("test", test)]
            dict_bar = tqdm(sets)
            for n, data in dict_bar:
                dict_bar.set_description(f"Setup Dictionary for split: {n}")
                # Setup dictionariy
                self._setup_dictionary(data, n)
            
            token_bar = tqdm(sets)
            # Tokenize text
            for n, data in token_bar:
                token_bar.set_description(f"Tokenize text for for split: {n}")
                setattr(self, n, tokenize(self.dictionary, data, n, self.ngrams, False, self.device))

    def display_text(self, t):
        for a in t:
            print(repr(self.dictionary.idx2word[a.item()]), end="")
        print()

    def display_list(self, l):
        for a in l:
            print(repr(self.dictionary.idx2word[a]), end="")
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

        with open(path, "r") as f:
            lines = f.readlines()

        self._setup_dictionary(lines, path)

    def _setup_dictionary(self, lines, label=""):
        token_frequency = Counter()

        for line in tqdm(lines, desc=f"Setup dictionary for {label}"):
            chars = ["<start>" for _ in range(1, self.ngrams)] + list(line) + ["<eos>"]
            for i in range(1, self.ngrams + 1):
                # Add UNK token for ngram
                n_unk_token = f"<{i}-UNK>"

                unk_idx = self.dictionary.add_word(n_unk_token)

                if unk_idx not in self.ngram_indexes[i]:
                    self.ngram_indexes[i].append(unk_idx)

                for ngram in ngrams(chars, i):
                    token_frequency["".join(ngram)] += 1

        self.dictionary.add_word("<start>")
        self.dictionary.add_word("<eos>")

        for toke, freq in token_frequency.items():
            if freq > self.unk_threshold or freq == -1:
                sanit_token = self.remove_marker_tokens(toke)
                idx = self.dictionary.add_word(toke)
                if idx not in self.ngram_indexes[len(sanit_token)]:
                    self.ngram_indexes[len(sanit_token)].append(idx)

    def tokenize_file(self, path):
        # Add words to the dictionary
        self.setup_dictionary(path)

        # Tokenize file content
        with open(path, "r", encoding="utf8") as f:
            lines = f.readlines()

        return tokenize(self.dictionary, lines, self.ngrams, path, False, self.device)

def tokenize_batch(dictionary, lines: List[str], ngram, label=None, otf=False, device="cpu"):
    """Tokenizes lines of text. Number of lines is already number of batches.

    Parameters
    ----------

    lines: List[str]
        List of strings, every string can represent a sentence or line of text.
    otf: bool
        On the Fly (oft) tokenization that leaves out the <eos> marker token,
        used for text generating of not complete sentence
    """


    n_gram_sequences = []

    padding_char_index = dictionary.get_idx_for_item(" ")

    len_longest_chunk: int = 0

    for n in range(1, ngram + 1):
        idss_n = []

        _lines = tqdm(lines, desc=f"Tokenize for {n}-gram sequence for {label}") if label else lines
        for line in _lines:

            # Adding start offsets for all ngrams
            words = ["<start>" for _ in range(1, n)]
            words.extend(list(line))
            if not otf:
                words.append("<eos>")

            ids = []
            length = 0
            for i, word in enumerate(ngrams(words, n)):
                try:
                    ids.append(dictionary.word2idx["".join(word)])
                except KeyError:
                    ids.append(dictionary.word2idx[f"<{n}-UNK>"])
                length += 1

            if len(ids) > len_longest_chunk:
                len_longest_chunk = len(ids)

            
            idss_n.append(ids)

        n_gram_sequences.append(idss_n)

    padded_char_sequence = []
    for i, ls in enumerate(n_gram_sequences):
        new_lines = []
        for j, line in enumerate(ls):
            line += [padding_char_index] * (len_longest_chunk - len(line))
            new_lines.append(torch.tensor(line).type(torch.int64))
        
        seq = torch.cat(new_lines).unsqueeze(dim=0)

        padded_char_sequence.append(seq)

    n_gram_sequences = torch.cat([torch.tensor(t) for t in padded_char_sequence]).to(
        device
    )

    return n_gram_sequences

def tokenize(dictionary, lines: List[str], ngram, label, otf=False, device="cpu"):
    """Tokenizes lines of text.

    Parameters
    ----------

    lines: List[str]
        List of strings, every string can represent a sentence or line of text.
    otf: bool
        On the Fly (oft) tokenization that leaves out the <eos> marker token,
        used for text generating of not complete sentence
    """

    n_gram_sequences = []
    min_length = sys.maxsize

    for n in range(1, ngram + 1):
        idss_n = []

        _lines = tqdm(lines, desc=f"Tokenize for {n}-gram sequence for {label}") if label else lines

        for line in _lines:

            # Adding start offsets for all ngrams
            words = ["<start>" for _ in range(1, n)]
            words.extend(list(line))
            if not otf:
                words.append("<eos>")

            ids = []
            length = 0
            for i, word in enumerate(ngrams(words, n)):
                try:
                    ids.append(dictionary.word2idx["".join(word)])
                except KeyError:
                    ids.append(dictionary.word2idx[f"<{n}-UNK>"])
                length += 1

            idss_n.append(torch.tensor(ids).type(torch.int64))
            
        # N-gram sequence, [1, #tokens]
        seq = torch.cat(idss_n).unsqueeze(dim=0)
        length = seq.size(1)

        if length < min_length:
            min_length = length

        n_gram_sequences.append(seq)

    n_gram_sequences = torch.cat([t[:min_length] for t in n_gram_sequences]).to(
        device
    )

    return n_gram_sequences

def grouped(iterable, n):
    # s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ...
    return zip(*[iter(iterable)] * n)

def prep_obw():
    import zipfile
    import requests

    r = requests.get("https://data.deepai.org/enwik8.zip", stream=True)
    
    with open("enwik8.zip", 'wb') as fd:
        for chunk in r.iter_content(chunk_size=128):
            fd.write(chunk)

    data = zipfile.ZipFile('enwik8.zip').read('enwik8')

def prep_enwiki8():
    # From: https://github.com/salesforce/awd-lstm-lm/blob/master/data/enwik8/prep_enwik8.py

    import os
    import sys
    import zipfile
    import requests


    if os.path.exists('data/enwik8/train.txt'):
        print('Tokenized enwik8 already exists - skipping processing')
        sys.exit()

    try:
        data = zipfile.ZipFile('enwik8.zip').read('enwik8')
    except:
        r = requests.get("https://data.deepai.org/enwik8.zip", stream=True)
        
        with open("enwik8.zip", 'wb') as fd:
            for chunk in r.iter_content(chunk_size=128):
                fd.write(chunk)

        data = zipfile.ZipFile('enwik8.zip').read('enwik8')

    print('Length of enwik8: {}'.format(len(data)))

    num_test_chars = 5000000

    train_data = data[: -2 * num_test_chars]
    valid_data = data[-2 * num_test_chars: -num_test_chars]
    test_data = data[-num_test_chars:]

    os.mkdir("data/enwik8")

    for fn, part in [('data/enwik8/train.txt', train_data), ('data/enwik8/valid.txt', valid_data), ('data/enwik8/test.txt', test_data)]:
        print('{} will have {} bytes'.format(fn, len(part)))
        print('- Tokenizing...')
        part_str = ' '.join([str(c) if c != ord('\n') else '\n' for c in part])
        print('- Writing...')
        f = open(fn, 'w').write(part_str)
        f = open(fn + '.raw', 'wb').write(part)

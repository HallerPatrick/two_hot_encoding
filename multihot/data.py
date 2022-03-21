import os
import sys
import json

from collections import Counter, defaultdict
from operator import itemgetter
from pathlib import Path
from typing import Dict, List, Optional, Union
from datasets.dataset_dict import DatasetDict

import torch
from colorama import init, Fore
from torch import Tensor

from nltk import ngrams
from tqdm import tqdm


class Dictionary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self._marker_tokens = []
        self.ngram_indexes = defaultdict(list)

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
            d = dict([(key, value) for key, value in self.word2idx.items()])
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

    valid: Optional[Tensor]
    test: Optional[Tensor]
    train: Optional[Tensor]

    def __init__(self, path, ngrams, unk_threshold, max_dict_size) -> None:
        self.unk_threshold = unk_threshold
        self.ngrams = ngrams
        self.max_dict_size = max_dict_size

        # Keep track of all indexes for each ngram, this is used
        # for the generating task
        self.ngram_indexes = defaultdict(list)

        self.dictionary = Dictionary()

        try:
            self.load_dataset_from_path(path)
        except FileNotFoundError as e:
            print(e)
            print("Try loading from huggingface")
            self.load_dataset_from_huggingface(path)

    def load_dataset_from_path(self, path: Union[str, Path]) -> None:

        if isinstance(path, str):
            path = Path(path)

        if not path.is_dir():
            raise FileNotFoundError(f"Directory: '{path}' cannot be found")

        train_files = ["train", "test", "valid"]

        train_to_lines: Dict[str, List[str]] = dict.fromkeys(train_files, [])

        # Setting up dictionary
        for train_split in train_files:

            # Check if paths exists
            train_file = Path(path) / (train_split + ".txt")
            assert train_file.is_file

            with open(str(train_file), "r") as f:
                lines = f.readlines()

            train_to_lines[train_split] = lines

        self.load_dataset(train_to_lines)

    def load_dataset_from_huggingface(self, path):

        from datasets.load import load_dataset

        name = path.split("/")

        # Load dataset
        dataset: DatasetDict = load_dataset(*name)

        train_files = {
            "train": dataset["train"]["text"],
            "valid": dataset["validation"]["text"],
            "test": dataset["test"]["text"],
        }

        self.load_dataset(train_files)

    def load_dataset(self, train_files: Dict[str, List[str]]):

        frequencies = Counter()

        # Setting up dictionary
        for train_split, lines in train_files.items():
            # Get token frequencies for every file
            file_frequency = self.get_frequency(lines, train_split)

            # Update global frequency count
            frequencies.update(file_frequency)

        # Populate dictionary
        self.setup_dictionary(frequencies)

        # Tokenizing
        for train_split, lines in train_files.items():
            tokenized_text = tokenize(
                self.dictionary, lines, self.ngrams, train_split, False
            )

            setattr(self, train_split, tokenized_text)

    def setup_dictionary(self, token_frequencies: Counter):

        with open("freqs.json", "w") as f:
            json.dump(dict(token_frequencies), f)

        self.dictionary.add_word("<start>")
        self.dictionary.add_word("<eos>")

        if self.max_dict_size > 0:
            for token, _ in token_frequencies.most_common(self.max_dict_size):
                sanit_token = self.remove_marker_tokens(token)
                idx = self.dictionary.add_word(token)
                if idx not in self.dictionary.ngram_indexes[len(sanit_token)]:
                    self.dictionary.ngram_indexes[len(sanit_token)].append(idx)
        else:
            for toke, freq in token_frequencies.items():
                if freq > self.unk_threshold or freq == -1:
                    sanit_token = self.remove_marker_tokens(toke)
                    idx = self.dictionary.add_word(toke)
                    if idx not in self.dictionary.ngram_indexes[len(sanit_token)]:
                        self.dictionary.ngram_indexes[len(sanit_token)].append(idx)

        print(f"Dictionary Size: {len(self.dictionary)}")

    def get_frequency(self, lines: List[str], train_split_label):
        token_frequency = Counter()

        for line in tqdm(
            lines,
            desc=f"Setup dictionary for {Fore.MAGENTA}{train_split_label}{Fore.RESET}",
        ):
            chars = ["<start>" for _ in range(1, self.ngrams)] + list(line) + ["<eos>"]
            for i in range(1, self.ngrams + 1):
                # Add UNK token for ngram
                n_unk_token = f"<{i}-UNK>"

                unk_idx = self.dictionary.add_word(n_unk_token)

                if unk_idx not in self.dictionary.ngram_indexes[i]:
                    self.dictionary.ngram_indexes[i].append(unk_idx)

                for ngram in ngrams(chars, i):
                    token_frequency["".join(ngram)] += 1

        return token_frequency

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


def tokenize_batch(
    dictionary, lines: List[str], ngram, label=None, otf=False
):
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

        _lines = (
            tqdm(lines, desc=f"Tokenize for {n}-gram sequence for {label}")
            if label
            else lines
        )
        for line in _lines:

            # Adding start offsets for all ngrams
            words = ["<start>" for _ in range(1, n)]
            words.extend(list(line))
            if not otf:
                words.append("<eos>")

            ids = []
            for word in ngrams(words, n):
                try:
                    ids.append(dictionary.word2idx["".join(word)])
                except KeyError:
                    ids.append(dictionary.word2idx[f"<{n}-UNK>"])

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

    n_gram_sequences = torch.cat([torch.tensor(t) for t in padded_char_sequence])
    return n_gram_sequences


def tokenize(dictionary, lines: List[str], ngram, label, otf=False):
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

        _lines = (
            tqdm(
                lines,
                desc=f"Tokenize for {n}-gram sequence for {Fore.GREEN}{label}{Fore.RESET}",
            )
            if label
            else lines
        )

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

    n_gram_sequences = torch.cat([t[:min_length] for t in n_gram_sequences])

    return n_gram_sequences


def grouped(iterable, n):
    # s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ...
    return zip(*[iter(iterable)] * n)


def prep_enwiki8():
    # From: https://github.com/salesforce/awd-lstm-lm/blob/master/data/enwik8/prep_enwik8.py

    import os
    import sys
    import zipfile
    import requests

    if os.path.exists("data/enwik8/train.txt"):
        print("Tokenized enwik8 already exists - skipping processing")
        sys.exit()

    try:
        data = zipfile.ZipFile("enwik8.zip").read("enwik8")
    except:
        r = requests.get("https://data.deepai.org/enwik8.zip", stream=True)

        with open("enwik8.zip", "wb") as fd:
            for chunk in r.iter_content(chunk_size=128):
                fd.write(chunk)

        data = zipfile.ZipFile("enwik8.zip").read("enwik8")

    print("Length of enwik8: {}".format(len(data)))

    num_test_chars = 5000000

    train_data = data[: -2 * num_test_chars]
    valid_data = data[-2 * num_test_chars : -num_test_chars]
    test_data = data[-num_test_chars:]

    os.mkdir("data/enwik8")

    for fn, part in [
        ("data/enwik8/train.txt", train_data),
        ("data/enwik8/valid.txt", valid_data),
        ("data/enwik8/test.txt", test_data),
    ]:
        print("{} will have {} bytes".format(fn, len(part)))
        print("- Tokenizing...")
        part_str = " ".join([str(c) if c != ord("\n") else "\n" for c in part])
        print("- Writing...")
        f = open(fn, "w").write(part_str)
        f = open(fn + ".raw", "wb").write(part)

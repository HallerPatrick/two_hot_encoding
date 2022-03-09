import math

from typing import List

import flair
import torch
import torch.nn as nn

from data import tokenize, tokenize_batch
from two_hot_encoding import NGramsEmbedding


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(
        self,
        dictionary, 
        nlayers: int,
        ngrams: int,
        hidden_size: int,
        unk_t: int,
        nout=None,
        embedding_size: int = 100,
        is_forward_lm = True,
        document_delimiter: str = '\n',
        dropout=0.1,
    ):
        super(RNNModel, self).__init__()

        self.ntoken = len(dictionary)

        self.encoder = NGramsEmbedding(len(dictionary), embedding_size)
        self.ngrams = ngrams
        self.unk_t = unk_t
        self.dictionary = dictionary
        self.nlayers = nlayers
        self.is_forward_lm = is_forward_lm
        self.nout = nout
        self.document_delimiter = document_delimiter
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.dropout = dropout

        if nlayers == 1:
            self.rnn = nn.LSTM(embedding_size, hidden_size, nlayers)
        else:
            self.rnn = nn.LSTM(embedding_size, hidden_size, nlayers, dropout=dropout)


        self.decoder = nn.Linear(hidden_size, len(dictionary))
        if nout is not None:
            self.proj = nn.Linear(hidden_size, nout)
            self.initialize(self.proj.weight)
            self.decoder = nn.Linear(nout, len(dictionary))
        else:
            self.proj = None
            self.decoder = nn.Linear(hidden_size, len(dictionary))

        self.init_weights()

    @staticmethod
    def initialize(matrix):
        in_, out_ = matrix.size()
        stdv = math.sqrt(3.0 / (in_ + out_))
        matrix.detach().uniform_(-stdv, stdv)

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, hidden):
        # [#ngram, #seq_len, #batch_size]
        emb = self.encoder(input)
        output, hidden = self.rnn(emb, hidden)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)

        return decoded, hidden

    def forward2(self, input, hidden, ordered_sequence_lengths=None):

        encoded = self.encoder(input)

        self.rnn.flatten_parameters()

        output, hidden = self.rnn(encoded, hidden)

        if self.proj is not None:
            output = self.proj(output)


        decoded = self.decoder(
            output.view(output.size(0) * output.size(1), output.size(2))
        )

        return (
            decoded.view(output.size(0), output.size(1), decoded.size(1)),
            output,
            hidden,
        )

    def init_hidden(self, bsz):
        weight = next(self.parameters()).detach()
        return (
            weight.new(self.nlayers, bsz, self.hidden_size).zero_().clone().detach(),
            weight.new(self.nlayers, bsz, self.hidden_size).zero_().clone().detach(),
        )

    def __getstate__(self):

        # serialize the language models and the constructor arguments (but nothing else)
        model_state = {
            "state_dict": self.state_dict(),

            "dictionary": self.dictionary,
            "is_forward_lm": self.is_forward_lm,
            "hidden_size": self.hidden_size,
            "nlayers": self.nlayers,
            "embedding_size": self.embedding_size,
            "nout": self.nout,
            "document_delimiter": self.document_delimiter,
            "dropout": self.dropout,
            "ngrams": self.ngrams,
            "unk_t": self.unk_t
        }

        return model_state

    def __setstate__(self, d):

        # special handling for deserializing language models
        if "state_dict" in d:

            # re-initialize language model with constructor arguments
            language_model = RNNModel(
                dictionary=d['dictionary'],
                nlayers=d['nlayers'],
                ngrams=d['ngrams'],
                hidden_size=d['hidden_size'],
                unk_t=d['unk_t'],
                nout=d['nout'],
                embedding_size=d['embedding_size'],
                is_forward_lm=d['is_forward_lm'],
                document_delimiter=d['document_delimiter'],
                dropout=d['dropout'],
            )

            language_model.load_state_dict(d['state_dict'])

            # copy over state dictionary to self
            for key in language_model.__dict__.keys():
                self.__dict__[key] = language_model.__dict__[key]

            # set the language model to eval() by default (this is necessary since FlairEmbeddings "protect" the LM
            # in their "self.train()" method)
            self.eval()

        else:
            self.__dict__ = d

    def save(self, file):
        model_state = {
            "state_dict": self.state_dict(),
            "dictionary": self.dictionary,
            "is_forward_lm": self.is_forward_lm,
            "hidden_size": self.hidden_size,
            "nlayers": self.nlayers,
            "embedding_size": self.embedding_size,
            "nout": self.nout,
            "document_delimiter": self.document_delimiter,
            "dropout": self.dropout,
            "ngrams": self.ngrams,
            "unk_t": self.unk_t
        }

        torch.save(model_state, str(file), pickle_protocol=4)

    def get_representation(
        self,
        strings: List[str],
        start_marker: str,
        end_marker: str,
        chars_per_chunk: int = 512,
    ):

        len_longest_str: int = len(max(strings, key=len))

        # pad strings with whitespaces to longest sentence
        padded_strings: List[str] = []

        for string in strings:
            if not self.is_forward_lm:
                string = string[::-1]

            padded = f"{start_marker}{string}{end_marker}"
            padded_strings.append(padded)

        # cut up the input into chunks of max charlength = chunk_size
        chunks = []
        splice_begin = 0
        longest_padded_str: int = len_longest_str + len(start_marker) + len(end_marker)
        for splice_end in range(chars_per_chunk, longest_padded_str, chars_per_chunk):
            chunks.append([text[splice_begin:splice_end] for text in padded_strings])
            splice_begin = splice_end

        chunks.append(
            [text[splice_begin:longest_padded_str] for text in padded_strings]
        )

        hidden = self.init_hidden(len(chunks[0]))

        batches: List[torch.Tensor] = []

        # push each chunk through the RNN language model
        for chunk in chunks:
            len_longest_chunk: int = len(max(chunk, key=len))
            sequences_as_char_indices: List[torch.Tensor] = []

            for string in chunk:
                chars = list(string) + [" "] * (len_longest_chunk - len(string))

                # [ngram, 1, sequence]
                n_gram_char_indices = tokenize_batch(self.dictionary, chars, self.ngrams, otf=True, device=flair.device).unsqueeze(dim=1)

                sequences_as_char_indices.append(n_gram_char_indices)
                
           
            # [ngram, batch_size, sequence]
            batches.append(torch.cat(sequences_as_char_indices, dim=1))
         
        output_parts = []
        for batch in batches:
            # [ngram, sequence, batch_size]
            batch = batch.transpose(1, 2)

            _, rnn_output, hidden = self.forward2(batch, hidden)

            output_parts.append(rnn_output)

        # concatenate all chunks to make final output
        output = torch.cat(output_parts)

        return output

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from two_hot_encoding import NGramsEmbedding


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(
        self,
        rnn_type: str,
        ntoken: int,
        ninp: int,
        nhid: int,
        nlayers: int,
        ngrams: int,
        unk_t: int,
        dropout=0.5,
        tie_weights=False,
    ):
        super(RNNModel, self).__init__()
        self.ntoken = ntoken
        self.encoder = NGramsEmbedding(ntoken, ninp)
        self.ngrams = ngrams
        self.unk_t = unk_t

        if rnn_type in ["LSTM", "GRU"]:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {"RNN_TANH": "tanh", "RNN_RELU": "relu"}[rnn_type]
            except KeyError:
                raise ValueError(
                    """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']"""
                )
            self.rnn = nn.RNN(
                ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout
            )

        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError(
                    "When using the tied flag, nhid must be equal to emsize"
                )
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, hidden):
        emb = self.encoder(input)
        output, hidden = self.rnn(emb, hidden)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        return decoded, hidden
        # return F.log_softmax(decoded, dim=1), hidden

    def forward_two_hot(self, input, input_two, hidden):
        emb = self.encoder(input, input_two)
        output, hidden = self.rnn(emb, hidden)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        return decoded, hidden
        # return F.log_softmax(decoded, dim=1), hidden

    def forward_char(self, input, hidden):
        emb = self.encoder(input)
        output, hidden = self.rnn(emb, hidden)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == "LSTM":
            return (
                weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid),
            )
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)


# class FlairNGramModel(LanguageModel):
#     def __init__(
#         self,
#         dictionary: Dictionary,
#         is_forward_lm: bool,
#         hidden_size: int,
#         nlayers: int,
#         embedding_size: int = 100,
#         nout=None,
#         document_delimiter: str = "\n",
#         dropout=0.1,
#     ):
#         super().__init__(
#             dictionary,
#             is_forward_lm,
#             hidden_size,
#             nlayers,
#             embedding_size=embedding_size,
#             nout=nout,
#             document_delimiter=document_delimiter,
#             dropout=dropout,
#         )

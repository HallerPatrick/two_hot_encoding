import math
import torch
import torch.nn as nn
import torch.nn.functional as F


from two_hot_encoding import TwoHotEmbedding

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.ntoken = ntoken
        # self.drop = nn.Dropout(dropout)
        self.encoder_uni = nn.Embedding(ntoken, ninp)
        self.encoder_bi = nn.Embedding(ntoken, ninp)
        # self.encoder = TwoHotEmbedding(ntoken, ninp)

        # self.comb_decoder = nn.Linear(ninp * 2, ninp)

        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights2(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder_uni.weight, -initrange, initrange)
        nn.init.uniform_(self.encoder_bi.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    # def forward(self, input, input_two, hidden):
    #     # emb = self.drop(self.encoder(input, input_two))
    #     emb = self.encoder(input, input_two)
    #     print(emb.size())
    #     # [35, 20, 200]
    #     output, hidden = self.rnn(emb, hidden)
    #     # output = self.drop(output)
    #     decoded = self.decoder(output)
    #     decoded = decoded.view(-1, self.ntoken)
    #     return decoded, hidden
    #     # return F.log_softmax(decoded, dim=1), hidden

    def forward(self, input, input_two, hidden):
        emb_uni = self.encoder_uni(input)
        emb_bi = self.encoder_bi(input_two)
        
        emb = torch.stack([emb_uni, emb_bi], dim=0).sum(dim=0)
       
        output, hidden = self.rnn(emb, hidden)
        # output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        return decoded, hidden



    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

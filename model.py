import math
import torch
import torch.nn as nn

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
        emb = self.encoder(input)
        output, hidden = self.rnn(emb, hidden)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        return decoded, hidden
        # return F.log_softmax(decoded, dim=1), hidden

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
        }

        torch.save(model_state, str(file), pickle_protocol=4)

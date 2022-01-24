from argparse import Namespace

import torch
from torch.utils.tensorboard import SummaryWriter

from itertools import product

from main import run_train
import data


def hparams(lr, batch_size, bptt, hidden_size, emsize, epochs):
    params = Namespace()
    params.data = "../../../Projects/character_bigrams/debug_data"
    # params.data = "./data/debug_data"
    params.only_unigrams = True
    params.seed = 1111
    params.cuda = False
    params.model = "LSTM"
    params.nlayers = 2
    params.dropout = 1.0
    params.tied = False
    params.clip = 0.25
    params.log_interval = 200
    params.dry_run = False
    params.save = "model.pt"
    params.is_transformer_model = False
    params.onnx_export = ""

    params.lr = lr
    params.batch_size = batch_size
    params.bptt = bptt
    params.nhid = hidden_size
    params.emsize = emsize
    params.epochs = epochs
    return params

def main():


    lr = [5]
    batch_size = [1, 5, 20, 50]
    bptt = [20]
    hidden_size = [50]
    emsize = [100]
    epochs = [10]
        
    combs = product(
        lr,
        batch_size,
        bptt, hidden_size, emsize,
        epochs
    )

    writer = SummaryWriter()

    for i, c in enumerate(combs):
        args = hparams(*c)
        run_train(args, writer=writer, no_run=i)

    writer.close()

def embs():
    
    device = "cpu"

    with open("model.pt", 'rb') as f:
        model = torch.load(f).to(device)

    corpus = data.Corpus("../../../Projects/character_bigrams/debug_data")
    writer = SummaryWriter()
    writer.add_embedding(model.encoder.embedding.weight, metadata=[repr(c)  for c in corpus.dictionary.word2idx.keys() ])
    writer.close()



if __name__ == "__main__":
    embs()

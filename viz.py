import torch

from torch.utils.tensorboard.writer import SummaryWriter

import data



with open("model.pt", 'rb') as f:
    model = torch.load(f)

model.eval()

corpus = data.Corpus("../../../Projects/character_bigrams/debug_data")
ntokens = len(corpus.dictionary)

writer = SummaryWriter("runs/testing")

writer.add_embedding(model.encoder.embedding.weight, metadata=corpus.dictionary.word2idx.keys(), tag="Embeddings")

writer.close()



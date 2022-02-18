###############################################################################
# Language Modeling on Wikitext-2
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import sys

import torch
from torch.nn import functional as F
from args import argparser_generate

import data
from main import batchify, get_batch

args = argparser_generate()


def get_best_ngrams(output, corpus, ngrams):

    best_ngrams = []

    for n in range(1, ngrams + 1):
        highest_prob = 0.0
        current_idx = -1

        for i, prob in enumerate(output.view(-1)):
            # For length comparison
            token = corpus.dictionary.idx2word[i]

            if token.startswith("<") and token.endswith(">"):
                token = "i"

            if len(token) == n:
                if prob > highest_prob:
                    highest_prob = prob
                    current_idx = i

        best_ngrams.append((current_idx, highest_prob))

    return best_ngrams


def display_prediction(prediction, corpus):
    prediction = F.softmax(prediction.view(-1), dim=0)
    preds = []
    for i, pred in enumerate(prediction):
        preds.append((i, pred.item()))

    preds = sorted(preds, key=lambda x: x[1], reverse=True)

    for p in preds:
        i, pred = p
        print("{:9}: {:.15f},".format(repr(corpus.dictionary.idx2word[i]), pred))


# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

with open(args.checkpoint, "rb") as f:
    model = torch.load(f).to(device)

model.eval()

corpus = data.Corpus(args.data, device, model.ngrams, model.unk_t)

ntokens = len(corpus.dictionary)

# Hidden with batch size 1
hidden = model.init_hidden(1)

# random first input
input = torch.randint(ntokens, (model.ngrams, 1, 1), dtype=torch.long).to(device)
# print(corpus.dictionary.word2idx)
# # input = torch.tensor(
# #     [[[corpus.dictionary.word2idx["T"]]], [[corpus.dictionary.word2idx["<start>T"]]]]
# # )
# input = torch.tensor([[[corpus.dictionary.word2idx["T"]]]])

generated_output = corpus.dictionary.idx2word[input[0][0].item()]

with open(args.outf, "w") as outf:
    with torch.no_grad():  # no tracking history
        for i in range(args.words):

            # print("+" * 89)
            hidden = model.init_hidden(1)
            output, hidden = model(input, hidden)
            output = output[-1]

            # display_prediction(output, corpus)
            output = F.softmax(output, dim=0)

            if args.temperature == 0.0:
                ngram_idx = torch.argmax(output)
            else:
                word_weights = output.squeeze().div(args.temperature).exp().cpu()

                # multinomial over all tokens
                ngram_idx = torch.multinomial(word_weights, 1)[0]

            word = corpus.dictionary.idx2word[ngram_idx]

            outf.write("({})".format(word))
            generated_output = generated_output + word

            print(f"{repr(generated_output)}", flush=True, end="")

            input = corpus.tokenize([generated_output], otf=True).unsqueeze(dim=2)

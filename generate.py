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

# random first input
input = torch.randint(ntokens, (model.ngrams, 1, 1), dtype=torch.long).to(device)

generated_output = corpus.dictionary.idx2word[input[0][0].item()]

with open(args.outf, "w") as outf:
    with torch.no_grad():  # no tracking history
        for i in range(args.words):

            # Reset hidden
            hidden = model.init_hidden(1)
            output, hidden = model(input, hidden)

            # Only use the generated ngrams
            output = output[-1]

            output = F.log_softmax(output, dim=0)

            if args.temperature == 0.0:
                # Just get highest confidence
                ngram_idx = torch.argmax(output)
            else:
                word_weights = output.squeeze().div(args.temperature).exp().cpu()

                # multinomial over all tokens
                ngram_idx = torch.multinomial(word_weights, 1)[0]

            # Get ngram word
            word = corpus.dictionary.idx2word[ngram_idx]

            # Append to generated sequence
            generated_output = generated_output + word

            # Use whole sequence as new input
            input = corpus.tokenize([generated_output], otf=True).unsqueeze(dim=2)

            # print(f"{repr(generated_output)}", flush=True, end="")
            outf.write("({})".format(word))

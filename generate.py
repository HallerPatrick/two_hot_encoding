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


def display_prediction(prediction, dictionary):
    prediction = F.softmax(prediction.view(-1), dim=0)
    preds = []
    for i, pred in enumerate(prediction):
        preds.append((i, pred.item()))

    preds = sorted(preds, key=lambda x: x[1], reverse=True)

    for p in preds:
        i, pred = p
        print("{:9}: {:.15f},".format(repr(dictionary.idx2word[i]), pred))


# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

with open(args.checkpoint, "rb") as f:
    model = torch.load(f).to(device)

model.eval()

ntokens = model.ntoken

# List of all indexes without UNK > 2 tokens
if model.ngrams > 2:
    unk_idxs = set(
        [model.dictionary.word2idx[f"<{i}-UNK>"] for i in range(3, model.ngrams + 1)]
    )
    token_idxs = set(model.dictionary.word2idx.values())
    token_idxs = torch.tensor(
        list(token_idxs.difference(unk_idxs)), dtype=torch.int64
    ).to(device)

# random first input
input = torch.randint(ntokens, (model.ngrams, 1, 1), dtype=torch.long).to(device)

generated_output = model.dictionary.idx2word[input[0][0].item()]

with open(args.outf, "w") as outf:
    with torch.no_grad():  # no tracking history
        for i in range(args.words):

            # Reset hidden
            hidden = model.init_hidden(1)
            print(input.size())
            output, hidden = model(input, hidden)

            print(hidden[0].size())

            # Only use the generated ngrams
            output = output[-1]

            if args.temperature == 0.0:
                output = F.softmax(output, dim=0).cpu()
                # Just get highest confidence
                ngram_idx = torch.argmax(output)
            else:
                output = F.log_softmax(output, dim=0)

                # Remove all UNK tokens for ngram > 2
                if model.ngrams > 2:
                    output = torch.index_select(output, 0, token_idxs).cpu()

                word_weights = output.squeeze().div(args.temperature).exp().cpu()

                # multinomial over all tokens
                ngram_idx = torch.multinomial(word_weights, 1)[0]

            # Get ngram word
            word = model.dictionary.idx2word[ngram_idx]

            # Append to generated sequence
            generated_output = generated_output + word

            # Use last 200 chars as sequence for new input
            input = data.tokenize(
                model.dictionary,
                [generated_output[-200:]],
                model.ngrams,
                label="Tokenize generated output",
                otf=True,
                device=device,
            ).unsqueeze(dim=2)

            outf.write(f"·{word}")

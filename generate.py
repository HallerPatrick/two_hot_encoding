###############################################################################
# Language Modeling on Wikitext-2
#
# This file generates new sentences sampled from the language model
#
###############################################################################

from pprint import pprint
import torch
from torch.nn import functional as F
from args import argparser_generate

import data
from two_hot_encoding import n_hot

args = argparser_generate()


def get_best(output, corpus, unigram=True, by_unigram=None):
    highest_prob = 0.0
    current_bigram = -1

    token_length = 1 if unigram else 2

    for i, prob in enumerate(output.view(-1)):
        # For length comparison
        token = (
            corpus.dictionary.idx2word[i].replace("<eos>", "i").replace("<start>", "i")
        )
        if len(token) == token_length:
            if prob > highest_prob:
                highest_prob = prob
                current_bigram = i

    if by_unigram:
        if corpus.dictionary.idx2word[current_bigram][0] == by_unigram:
            return current_bigram, highest_prob
        return None
    else:
        return current_bigram, highest_prob


def get_best_ngrams(output, corpus, ngrams):

    best_ngrams = []
    for n in range(1, ngrams + 1):
        highest_prob = 0.0
        current_idx = -1

        for i, prob in enumerate(output.view(-1)):
            # For length comparison
            token = (
                corpus.dictionary.idx2word[i]
                .replace("<eos>", "i")
                .replace("<start>", "i")
            )
            if len(token) == n:
                if prob > highest_prob:
                    highest_prob = prob
                    current_idx = i

        best_ngrams.append((current_idx, highest_prob))

    return best_ngrams


def display_prediction(prediction, corpus):
    prediction = F.softmax(prediction.view(-1))
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

if args.temperature < 1e-3:
    print("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, "rb") as f:
    model = torch.load(f).to(device)

model.eval()

corpus = data.Corpus(args.data, args.only_unigrams, model.ngrams, args.unk_t)

pprint(corpus.ngram_indexes)
ntokens = len(corpus.dictionary)

is_transformer_model = (
    hasattr(model, "model_type") and model.model_type == "Transformer"
)

if not is_transformer_model:
    hidden = model.init_hidden(1)

input = torch.randint(ntokens, (model.ngrams, 1, 1), dtype=torch.long).to(device)

with open(args.outf, "w") as outf:
    with torch.no_grad():  # no tracking history
        for i in range(args.words):
            if is_transformer_model:
                output = model(input, False)
                word_weights = output[-1].squeeze().div(args.temperature).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                word_tensor = torch.Tensor([[word_idx]]).long().to(device)
                input = torch.cat([input, word_tensor], 0)
                word = corpus.dictionary.idx2word[word_idx]
            else:

                output, hidden = model(input, hidden)

                display_prediction(output, corpus)
                print("+" * 89)
                output = F.softmax(output, dim=1)

                word_weights = output.squeeze().div(args.temperature).exp().cpu()

                ngram_idxs = []
                # Iter over all ngram idxs from corpus
                for ngram, idxs in corpus.ngram_indexes.items():

                    # Select indexes from output distribution
                    ngram_word_weights = torch.index_select(
                        word_weights, 0, torch.tensor(sorted(idxs))
                    )

                    # Get best ngram idx
                    ngram_idx = torch.multinomial(ngram_word_weights, 1)[0]

                    # Trace back the original idx
                    for i, idx in enumerate(sorted(idxs)):
                        if ngram_idx.item() == i:
                            ngram_idxs.append(torch.tensor([[idx]]).unsqueeze(dim=0))
                            break

                # Build new input from best ngram indexes
                input = torch.cat(ngram_idxs)

                best_ngrams = get_best_ngrams(output, corpus, args.ngrams)

                word = corpus.dictionary.idx2word[best_ngrams[0][0]]

            outf.write("({})".format(word))  # + ('\n' if i % 100 == 99 else ''))

            if i % args.log_interval == 0:
                print("| Generated {}/{} words".format(i, args.words))

###############################################################################
# Language Modeling on Wikitext-2
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse

import torch
from torch.nn import functional as F

import data

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=2.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
parser.add_argument('--only-unigrams', action='store_true',
                    help='use character based language model')
args = parser.parse_args()


def get_best(output, corpus, unigram=True, by_unigram=None):
    highest_prob = 0.0
    current_bigram = -1

    token_length = 1 if unigram else 2

    for i, prob in enumerate(output.view(-1)):
        if len(corpus.dictionary.idx2word[i]) == token_length:
            if prob > highest_prob:
                highest_prob = prob
                current_bigram = i

    if by_unigram:
        if corpus.dictionary.idx2word[current_bigram][0] == by_unigram:
            return current_bigram
        return None
    else:
        return current_bigram


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
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f).to(device)
model.eval()

corpus = data.Corpus(args.data, args.only_unigrams)
ntokens = len(corpus.dictionary)

is_transformer_model = hasattr(model, 'model_type') and model.model_type == 'Transformer'
if not is_transformer_model:
    hidden = model.init_hidden(1)

input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)
input_two = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)

last_word = corpus.dictionary.idx2word[input]

with open(args.outf, 'w') as outf:
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

                if args.only_unigrams:
                    output, hidden = model(input, hidden)
                else:
                    output, hidden = model(input, input_two, hidden)

                display_prediction(output, corpus)
                output = F.softmax(output, dim=1)

                if args.only_unigrams:
                    # Get best unigram
                    word_idx = get_best(output, corpus)
                    input.fill_(word_idx)
                    word = corpus.dictionary.idx2word[word_idx]
                else:
                    # Get best unigram
                    word_idx = get_best(output, corpus)
                    input.fill_(word_idx)
                    word = corpus.dictionary.idx2word[word_idx]
                    print(f"Unigram: {word}")

                    best_bigram_idx = get_best(output, corpus, False)

                    fitting_bigram_idx = get_best(output, corpus, False, by_unigram=word)

                    # If we found a suitable bigram
                    if fitting_bigram_idx:
                        # If last prediction was a unigram
                        if len(last_word) == 1:
                            word = corpus.dictionary.idx2word[fitting_bigram_idx]
                            print(f"Bigram: {word}")

                    input_two.fill_(best_bigram_idx)

                print()
                print(">>>>", repr(word))
                print()

            last_word = word
            outf.write("({})".format(word))  # + ('\n' if i % 100 == 99 else ''))

            if i % args.log_interval == 0:
                print('| Generated {}/{} words'.format(i, args.words))


def gen(model, corpus, device):
    output_string = []

    hidden = model.init_hidden(1)

    ntokens = len(corpus.dictionary)

    input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)

    with torch.no_grad():  # no tracking history
        for i in range(1000):
            output, hidden = model(input, hidden)

            word_weights = output.squeeze().div(1.0).exp().cpu()
            # unigram_idx, unigram_prob = get_idx(word_weights, corpus.dictionary.idx2word)
            word_idx = torch.multinomial(word_weights, 1)[0]
            input.fill_(word_idx)

            # word_idx = torch.multinomial(word_weights, 1)[0]

            unigram = corpus.dictionary.idx2word[word_idx]

            word = unigram

            output_string.append(word)

    return "".join(output_string)

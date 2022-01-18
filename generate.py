###############################################################################
# Language Modeling on Wikitext-2
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse

import torch

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
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
args = parser.parse_args()


def get_idx(word_weights, dictionary, unigram=True):
    current_idx = -1
    current_prob = 0.0
    token_length = 1 if unigram else 2

    for i, prob in enumerate(word_weights):
        if len(dictionary[i]) == token_length:
            if prob > current_prob:
                current_idx = i
                current_prob = prob

    return current_idx, current_prob

def debug_prediction(predictions: torch.Tensor, corpus, top_k: int = 3, unigrams=True):
    """Get a debug representation of the prediction"""

    # probs = torch.nn.functional.softmax(predictions, dim=1)
    results = torch.topk(predictions, top_k)
    results = results[1]

    token_length = 1 if unigrams else 2
    tries = 0
    
    print("Resulting predictions:")
    for result in results:
        print("Next tokens: ", end="")
        i = 0
        while i != top_k:
            if tries == len(predictions[0]):
                break
            idx = result[i]
            word = corpus.dictionary.idx2word[idx.item()]
            if len(word) == token_length:
                print(f"{idx} -> {repr(word)}({probs[0][idx]}), ", end="")
                i += 1
            tries += 1
        print()

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

corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)

is_transformer_model = hasattr(model, 'model_type') and model.model_type == 'Transformer'
if not is_transformer_model:
    hidden = model.init_hidden(1)

input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)
input_two = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)

with open(args.outf, 'w') as outf:
    with torch.no_grad():  # no tracking history
        for i in range(args.words):
            output, hidden = model(input, input_two, hidden)
            word_weights = output.squeeze().div(args.temperature).exp().cpu()
            unigram_idx, unigram_prob = get_idx(word_weights, corpus.dictionary.idx2word)
            bigram_idx, bigram_prob = get_idx(word_weights, corpus.dictionary.idx2word, unigram=False)
            # word_idx = torch.multinomial(word_weights, 1)[0]
            input.fill_(unigram_idx)
            input_two.fill_(bigram_idx)


            unigram = corpus.dictionary.idx2word[unigram_idx]
            bigram = corpus.dictionary.idx2word[bigram_idx]

            word = None            
            if unigram_prob > bigram_prob:
                word = unigram
            else:
                word = bigram
            
            # debug_prediction(output, corpus, top_k=10, unigrams=False)

            outf.write("{}".format(word) + ('\n' if i % 20 == 19 else ''))

            if i % args.log_interval == 0:
                print('| Generated {}/{} words'.format(i, args.words))

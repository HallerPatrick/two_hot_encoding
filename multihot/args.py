import argparse
from argparse import Namespace

from prettytable.prettytable import PrettyTable

import yaml

def read_config(path):
    """Return namespace object like argparser of yaml file"""

    with open(path, "r") as f:
        conf = yaml.safe_load(f)
    
    table = PrettyTable([ "Parameter", "Value" ])
    
    for parameter, value in conf.items():
        table.add_row([parameter, value])
    
    print("Configrurations:")
    print(table)
    return Namespace(**conf)

def argparser_generate():

    parser = argparse.ArgumentParser(description="PyTorch Wikitext-2 Language Model")

    # Model parameters.
    parser.add_argument(
        "--data",
        type=str,
        default="./data/wikitext-2",
        help="location of the data corpus",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="./model.pt", help="model checkpoint to use"
    )
    parser.add_argument(
        "--outf",
        type=str,
        default="generated.txt",
        help="output file for generated text",
    )
    parser.add_argument(
        "--words", type=int, default="1000", help="number of words to generate"
    )
    parser.add_argument(
        "--unk-t", type=int, default=3, help="UNK threshold for bigrams"
    )
    parser.add_argument("--ngrams", type=int, default=2, help="N-Grams used")
    parser.add_argument("--seed", type=int, default=1111, help="random seed")
    parser.add_argument("--cuda", action="store_true", help="use CUDA")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="temperature - higher will increase diversity",
    )
    parser.add_argument(
        "--log-interval", type=int, default=100, help="reporting interval"
    )
    return parser.parse_args()


def argparser_train():

    parser = argparse.ArgumentParser(
        description="PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model"
    )
    parser.add_argument("--config", type=str, default="", help="Configration file (YAML) for all arguments, if empty, use command lines arguments")
    parser.add_argument(
        "--data",
        type=str,
        default="./data/wikitext-2",
        help="location of the data corpus",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="LSTM",
        help="type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)",
    )
    parser.add_argument(
        "--unk-t", type=int, default=3, help="UNK threshold for bigrams"
    )
    parser.add_argument(
        "--max-dict-size",
        type=int,
        default=0,
        help="Set max dictionary size, other tokens become UNK",
    )
    parser.add_argument("--ngrams", type=int, default=2, help="N-Grams used")
    parser.add_argument(
        "--unk-fallback",
        action="store_true",
        help="Fallback on n-1-gram if UNK for n-gram",
    )
    parser.add_argument(
        "--unigram-ppl",
        action="store_true",
        help="Calculate perplexity only over unigrams",
    )
    parser.add_argument(
        "--emsize", type=int, default=200, help="size of word embeddings"
    )
    parser.add_argument(
        "--nhid", type=int, default=200, help="number of hidden units per layer"
    )
    parser.add_argument("--nlayers", type=int, default=2, help="number of layers")
    parser.add_argument("--lr", type=float, default=20, help="initial learning rate")
    parser.add_argument("--clip", type=float, default=0.25, help="gradient clipping")
    parser.add_argument("--epochs", type=int, default=40, help="upper epoch limit")
    parser.add_argument(
        "--batch_size", type=int, default=20, metavar="N", help="batch size"
    )
    parser.add_argument("--bptt", type=int, default=35, help="sequence length")
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="dropout applied to layers (0 = no dropout)",
    )
    parser.add_argument(
        "--tied", action="store_true", help="tie the word embedding and softmax weights"
    )
    parser.add_argument("--seed", type=int, default=1111, help="random seed")
    parser.add_argument("--cuda", action="store_true", help="use CUDA")
    parser.add_argument(
        "--log-interval", type=int, default=200, metavar="N", help="report interval"
    )
    parser.add_argument(
        "--save", type=str, default="model.pt", help="path to save the final model"
    )
    parser.add_argument(
        "--onnx-export",
        type=str,
        default="",
        help="path to export the final model in onnx format",
    )
    parser.add_argument(
        "--nhead",
        type=int,
        default=2,
        help="the number of heads in the encoder/decoder of the transformer model",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="verify the code and the model"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        help="optimizer for training [adam, sgd]",
    )

    parser.add_argument(
        "--wdecay",
        type=float,
        default=1.2e-6,
        help="weight decay applied to all weights",
    )

    return parser.parse_args()

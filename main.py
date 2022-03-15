# coding: utf-8
import math
import time
from typing import Optional

import torch

import data
import model as _model

from args import argparser_train
from loss import CrossEntropyLossSoft
from torch_utils import (
    batchify,
    count_parameters,
    export_onnx,
    get_batch,
    repackage_hidden,
)
from two_hot_encoding import soft_n_hot


device: Optional[str] = None


def run_train(args):
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print(
                "WARNING: You have a CUDA device, so you should probably run with --cuda"
            )

    device = torch.device("cuda" if args.cuda else "cpu")

    ###############################################################################
    # Load data
    ###############################################################################

    corpus = data.Corpus(args.data, device, args.ngrams, args.unk_t, args.max_dict_size)

    print(f"Dictionary Size: {len(corpus.dictionary)}")

    eval_batch_size = 10
    train_data = batchify(corpus.train, args.batch_size, device)
    val_data = batchify(corpus.valid, eval_batch_size, device)
    test_data = batchify(corpus.test, eval_batch_size, device)

    ###############################################################################
    # Build the model
    ###############################################################################
    if args.model == "Transformer":
        model = _model.TransformerModel(
            corpus.dictionary,
            args.emsize,
            args.nhead,
            args.nhid,
            args.nlayers,
            args.ngrams,
            args.unk_t,
            args.dropout,
        ).to(device)
    else:
        model = _model.RNNModel(
            corpus.dictionary,
            args.nlayers,
            args.ngrams,
            args.nhid,
            args.unk_t,
            None,
            args.emsize,
            dropout=args.dropout,
        ).to(device)

    count_parameters(model)

    # TODO: Weighted loss labels?
    weights = None  # torch.ones((ntokens))
    # for n, n_idxs in corpus.ngram_indexes.items():
    #     for idxs in n_idxs:
    #         weights[idxs] = n

    criterion = CrossEntropyLossSoft(weight=weights)

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.wdecay
        )
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, weight_decay=args.wdecay
        )

    ###############################################################################
    # Training code
    ###############################################################################
    def evaluate(data_source):
        # Turn on evaluation mode which disables dropout.
        model.eval()
        total_loss = 0.0
        ntokens = len(corpus.dictionary)

        if args.model != "Transformer":
            hidden = model.init_hidden(eval_batch_size)

        with torch.no_grad():
            for i in range(0, data_source.size(1) - args.ngrams, args.bptt):
                data, targets = get_batch(data_source, i, args.bptt)
                targets = soft_n_hot(targets, ntokens)

                if args.model == "Transformer":
                    output = model(data)
                    output = output.view(-1, ntokens)

                else:
                    output, hidden = model(data, hidden)
                    hidden = repackage_hidden(hidden)

                print(output.size())

                # Perplexity only based on unigram candidates
                if args.unigram_ppl:
                    output = torch.index_select(
                        output,
                        1,
                        torch.tensor(corpus.ngram_indexes[1]).to(output.device),
                    )

                    targets = torch.index_select(
                        targets,
                        1,
                        torch.tensor(corpus.ngram_indexes[1]).to(targets.device),
                    )
                    total_loss += len(data[0]) * criterion(output, targets).item()
                else:
                    total_loss += len(data[0]) * criterion(output, targets).item()

        return total_loss / (len(data_source[0]) - 1)

    def train():
        # Turn on training mode which enables dropout.
        model.train()
        total_loss = 0.0
        start_time = time.time()
        ntokens = len(corpus.dictionary)
        if args.model != "Transformer":
            hidden = model.init_hidden(args.batch_size)
        for batch, i in enumerate(
            range(0, train_data.size(1) - args.ngrams, args.bptt)
        ):
            data, targets = get_batch(train_data, i, args.bptt)

            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            model.zero_grad()
            if args.model == "Transformer":
                output = model(data)
                output = output.view(-1, ntokens)
            else:
                hidden = repackage_hidden(hidden)

                output, hidden = model(data, hidden)

            targets = soft_n_hot(targets, ntokens)

            loss = criterion(output, targets)
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            for p in model.parameters():
                p.data.add_(p.grad, alpha=-optimizer.param_groups[0]["lr"])

            optimizer.step()
            total_loss += loss.item()

            if batch % args.log_interval == 0 and batch > 0:
                cur_loss = total_loss / args.log_interval
                elapsed = time.time() - start_time
                print(
                    "| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.2f} | "
                    "loss {:5.2f} | ppl {:8.2f} | bpc {:8.2f}".format(
                        epoch,
                        batch,
                        len(train_data[0]) // args.bptt,
                        optimizer.param_groups[0]["lr"],
                        elapsed * 1000 / args.log_interval,
                        cur_loss,
                        math.exp(cur_loss),
                        cur_loss / math.log(2),
                    )
                )
                total_loss = 0
                start_time = time.time()
            if args.dry_run:
                break

    # Loop over epochs.
    best_val_loss = None

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train()
            val_loss = evaluate(val_data)

            print("-" * 96)
            print(
                "| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | "
                "valid ppl {:8.2f} | valid bpc {:8.2f}".format(
                    epoch,
                    (time.time() - epoch_start_time),
                    val_loss,
                    math.exp(val_loss),
                    val_loss / math.log(2),
                )
            )
            print("-" * 96)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(args.save, "wb") as f:
                    torch.save(model, f)

                model.save("flair_" + args.save)

                best_val_loss = val_loss
            else:
                optimizer.param_groups[0]["lr"] /= 10.0

    except KeyboardInterrupt:
        print("-" * 89)
        print("Exiting from training early")

    # Load the best saved model.
    with open(args.save, "rb") as f:
        model = torch.load(f).to(device)

        # after load the rnn params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        # Currently, only rnn model supports flatten_parameters function.
        if args.model in ["RNN_TANH", "RNN_RELU", "LSTM", "GRU"]:
            model.rnn.flatten_parameters()

    test_loss = evaluate(test_data)

    print("=" * 89)
    print(
        "| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.2f}".format(
            test_loss, math.exp(test_loss), test_loss / math.log(2)
        )
    )
    print("=" * 89)

    if len(args.onnx_export) > 0:
        # Export the model in ONNX format.
        export_onnx(
            model, args.onnx_export, batch_size=1, seq_len=args.bptt, device=device
        )


if __name__ == "__main__":
    args = argparser_train()
    run_train(args)

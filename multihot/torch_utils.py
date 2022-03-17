"""
Some necessary and unnecessary utilities
"""

import torch

from prettytable import PrettyTable


def get_batch(source, i, bptt, device):
    """
    get_batch subdivides the source data into chunks of length args.bptt.
    If source is equal to the example output of the batchify function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM.
    """
    ngrams = source.size(0)
    seq_len = min(bptt, source.size(1) - ngrams - i)

    # [ngram, sequnces, bsz]
    data = source[:, i : i + seq_len].to(device)

    targets = []
    for ngram in range(1, ngrams + 1):
        target = source[ngram - 1, i + ngram : i + ngram + seq_len]
        targets.append(target.view(-1).unsqueeze(dim=0))

    targets = torch.cat(targets).to(device)

    return data, targets


def batchify(data, bsz, device):
    """
    Starting from sequential data, batchify arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.
    These columns are treated as independent by the model, which means that the
    dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
    batch processing.
    Work out how cleanly we can divide the dataset into bsz parts.
    """
    ngrams = data.size(0)
    nbatch = data.size()[-1] // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(1, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(ngrams, bsz, -1)
    data = torch.transpose(data, 1, 2).contiguous()
    return data.to(device)


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def export_onnx(model, path, batch_size, seq_len, device):
    model.eval()
    dummy_input = (
        torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    )
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)


def display_input_n_gram_sequences(input, corpus):
    print("Target sequences:")
    for i in input.size()[0]:
        print(f"{i}-gram")
        corpus.display_text(input[i][:, :1])


def display_target_n_gram_sequences(input, corpus, batch_size):
    print("Target sequences:")
    for i in input.size()[0]:
        print(f"{i}-gram")
        corpus.display_text(input[i][::batch_size])


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

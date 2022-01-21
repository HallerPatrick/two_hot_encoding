from argparse import Namespace

from torch.utils.tensorboard import SummaryWriter

from itertools import product

from main import run_train


def hparams(lr, batch_size, bptt, hidden_size, emsize, epochs):
    params = Namespace()
    params.data = "./data/short"
    params.only_unigrams = True
    params.seed = 1111
    params.cuda = False
    params.model = "LSTM"
    params.nlayers = 2
    params.dropout = 1.0
    params.tied = False
    params.clip = 0.25
    params.log_interval = 200
    params.dry_run = False
    params.save = "model.pt"
    params.is_transformer_model = False
    params.onnx_export = ""

    params.lr = lr
    params.batch_size = batch_size
    params.bptt = bptt
    params.nhid = hidden_size
    params.emsize = emsize
    params.epochs = epochs
    return params

def main():


    lr = [0.01, 0.1, 5, 10]
    batch_size = [1, 5, 20, 50]
    bptt = [1, 20, 50, 100]
    hidden_size = [10, 50, 100]
    emsize = [50, 100, 200]
    epochs = [1, 10, 50]
        
    combs = product(
        lr,
        batch_size,
        bptt, hidden_size, emsize,
        epochs
    )

    writer = SummaryWriter()

    for i, c in enumerate(combs):
        args = hparams(*c)
        run_train(args, writer=writer, no_run=i)

    writer.close()


if __name__ == "__main__":
    main()

# N-Gram Character Level Language Model

### Improving Language Models by Jointly Modelling Language as Distributions over Characters and Bigrams

More information on this project on my [website](https://hallerpatrick.github.io/) and in the [expose](./expose.pdf)


## Disclaimer

This is a WIP and will not accept PRs for now.


## Measurments

### Wikitext-2

| Model | Emebdding Size | Epochs | Learning Rate | Hidden Size | Embedding Size | Batch Size | Dataset | NGram | Test PPL | Test BTC 
| Cl LSTM | 128 | 30 | 20 | 128 | 128 | 50 | Wikitext-2 | 1 | 3.76 | 1.91
| N-Gram CL LSTM       | 128 | 30 | 20 | 128 | 128 | 50 | Wikitext-2 | 1 | 3.72 | 1.89
| N-Gram CL LSTM       | 128 | 30 | 20 | 128 | 128 | 50 | Wikitext-2 | 2 | 11.65 | 1.86
| N-Gram CL LSTM       | 128 | 30 | 20 | 128 | 128 | 50 | Wikitext-2 | 2 | 

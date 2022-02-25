# N-Gram Character Level Language Model

### Improving Language Models by Jointly Modelling Language as Distributions over Characters and Bigrams

More information on this project on my [website](https://hallerpatrick.github.io/) and in the [expose](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/a3d605bc-4314-4101-937b-5b8763421361/Expose_NLP-6.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220221%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220221T173652Z&X-Amz-Expires=86400&X-Amz-Signature=45dd6b2619d195482c955e809fb7ba0cd167e4d442c6c425b5050b7b01e22926&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Expose_NLP-6.pdf%22&x-id=GetObject)


## Disclaimer

This is a WIP and will not accept PRs for now.

## Measurments

### Benchmarks

| Model | Emebdding Size | Hidden Size | BPTT | Batch Size | Epochs | Layers | Dataset | LR | NGram | Test PPL | Test BPC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CL LSTM | 128 | 128  | 35  | 50 | 30 | 2 | Wikitext-2 | 20 (1/4 decay) | 1 | 3.76 | 1.91
| N-Gram CL LSTM       | 128 | 128  | 35  | 50 | 30 | 2 | Wikitext-2 | 20 (1/4 decay) | 1 | 3.72 | 1.89
| N-Gram CL LSTM       | 128 | 128  | 35  | 50 | 30 | 2 | Wikitext-2 | 20 (1/4 decay) | 2 | 11.72 | 8.12
| N-Gram CL LSTM       | 128 | 128  | 35  | 50 | 30 | 2 | Wikitext-2 | 20 (1/4 decay) | 2 | 1.96 (only unigrams) | 0.47 (only unigrams)
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N-Gram CL LSTM       | 512 | 512 | 200 | 50 | 34 | 2 | Wikitext-103 | 20 (1/4 decay) | 2 | 7.96 | 2.98
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N-Gram CL LSTM       | 400 | 1840 | 200 | 128 | 23 | 3 | enwiki-8 | 10 (1/4 decay) | 2 | 1.63 | 0.69
| Paper LSTM [^1] [^2] | 400 | 1840 | 200 | 128 | 50 | 3 | enwiki-8 | 0.001 (1/10 decay) | 1 | - | 1.232
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N-Gram CL LSTM       | 200 | 1000 | 150 | 128 | 500 | 3 | ptb | 4 (1/4 decay) | 2 | 8.01 | 3.00
| N-Gram CL LSTM       | 200 | 1000 | 150 | 128 | 500 | 3 | ptb | 4 (1/4 decay) | 2 | 1.60 (only unigrams) | 0.68 (only unigrams) (no optimizer)
| N-Gram CL LSTM       | 200 | 1000 | 150 | 128 | 69 | 3 | ptb | 0.002 | 1.56 (only unigrams) | 0.64 (only unigrams) 
| Paper LSTM [^1] [^2] | 400 | 1000 | 150 | 128 | 500 | 3 | ptb | 0.001 (1/10 decay) | 1 | 1.232



### Model and Corpus train/size metrics


Investigate the dictionary size, the models size aswell as memory usage and training time
for different n-grams and UNK threshold

_Hyperparamter_:
```
Hidden size: 128
emsize: 128
nlayers: 2
batch size: 50
lr: 20
```


| Dataset | UNK  | Ngram | sec/epoch | dictionary size | GPU memory usage | Params |
| ---     | ---  | ---   | ---       | ---             | ---              | ---    |
| Wiki-2  | 0    | 1     | 43        | 1156            | 2553 MiB         | 561284 
| Wiki-2  | 5    | 2     | 67        | 2754            | 2817 MiB         | 971713
| Wiki-2  | 20   | 2     | 53        | 1751            | 2761 MiB         | 714199
| Wiki-2  | 40   | 2     | 48        | 1444            | 2763 MiB         | 635300
| Wiki-2  | 40   | 3     | 143       | 9827            | 3301 MiB         | 2789721
| Wiki-2  | 40   | 4     | 412       | 34124           | 4723 MiB         | 9034060
| Wiki-2  | 40   | 5     | 782       | 72745           | 6765  MiB        | 18959657
| Wiki-2  | 100  | 10    | 965       | 90250           | 8635  MiB        | 23458442
| ptb  | 20   | 2     | 31         | 704           | 4893 MiB      | 21669504
| enwik8  | 20   | 2     | 4493         |            |  13 GiB      | 120M

### Plot for Wikitet-2

![Memory usage and model size](./table1.png)


[^1]: https://arxiv.org/pdf/1803.08240.pdf
[^2]: https://github.com/salesforce/awd-lstm-lm

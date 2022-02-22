# N-Gram Character Level Language Model

### Improving Language Models by Jointly Modelling Language as Distributions over Characters and Bigrams

More information on this project on my [website](https://hallerpatrick.github.io/) and in the [expose](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/a3d605bc-4314-4101-937b-5b8763421361/Expose_NLP-6.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220221%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220221T173652Z&X-Amz-Expires=86400&X-Amz-Signature=45dd6b2619d195482c955e809fb7ba0cd167e4d442c6c425b5050b7b01e22926&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Expose_NLP-6.pdf%22&x-id=GetObject)


## Disclaimer

This is a WIP and will not accept PRs for now.


## Measurments

### Wikitext-2

| Model | Emebdding Size | Hidden Size | BPTT | Batch Size | Epochs | Layers | Dataset | LR | NGram | Test PPL | Test BPC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Character Level LSTM | 128 | 128  | 35  | 50 | 30 | 2 | Wikitext-2 | 20 (1/4 decay) | 1 | 3.76 | 1.91
| N-Gram CL LSTM       | 128 | 128  | 35  | 50 | 30 | 2 | Wikitext-2 | 20 (1/4 decay) | 1 | 3.72 | 1.89
| N-Gram CL LSTM       | 128 | 128  | 35  | 50 | 30 | 2 | Wikitext-2 | 20 (1/4 decay) | 2 | 11.72 | 8.12
| N-Gram CL LSTM       | 128 | 128  | 35  | 50 | 30 | 2 | Wikitext-2 | 20 (1/4 decay) | 2 | 1.96 (only unigrams) | 0.47 (only unigrams)
| N-Gram CL LSTM       | 512 | 512 | 200 | 50 | 34 | 2 | Wikitext-103 | 20 (1/4 decay) | 2 | 7.96 | 2.98
| N-Gram CL LSTM       | 400 | 1840 | 200 | 128 | 50 | 3 | enwiki-8 | 0.001 (1/10 decay) | 2 |

TODO:
    Source: https://github.com/salesforce/awd-lstm-lm
    ``` 
    python -u main.py --epochs 50 --nlayers 3 --emsize 400 --nhid 1840 --alpha 0 --beta 0 --dropoute 0 --dropouth 0.1 --dropouti 0.1 --dropout 0.4 --wdrop 0.2 --wdecay 1.2e-6 --bptt 200 --batch_size 128 --optimizer adam --lr 1e-3 --data data/enwik8 --save ENWIK8.pt --when 25 35
    ```


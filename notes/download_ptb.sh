echo "- Downloading Penn Treebank (PTB)"
wget --quiet --continue http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
tar -xzf simple-examples.tgz

mkdir -p data/pennchar

echo "Saved in data/pennchar"
mv simple-examples/data/ptb.train.txt data/pennchar/train.txt
mv simple-examples/data/ptb.test.txt data/pennchar/test.txt
mv simple-examples/data/ptb.valid.txt data/pennchar/valid.txt

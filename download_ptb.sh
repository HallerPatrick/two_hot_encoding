echo "- Downloading Penn Treebank (PTB)"
wget --quiet --continue http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
tar -xzf simple-examples.tgz

echo "- Downloading Penn Treebank (Character)"
mkdir -p data/pennchar

mv simple-examples/data/ptb.char.train.txt data/pennchar/train.txt
mv simple-examples/data/ptb.char.test.txt data/pennchar/test.txt
mv simple-examples/data/ptb.char.valid.txt data/pennchar/valid.txt

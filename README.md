# SNUTOK test repository with transformer translation task

> Experiment Space for Cho

> **- Note that post-slack commands in the scripts won't work. Do not use the scripts without modification.**

## Write Codes

+ Implement a tokenizer class that extends [BaseTokenizer](tokenizer/base.py)
+ Write a tokenizer training script for a tokenizer. [Reference](scripts/build_char_vocab.py)

## Prepare storage

> Checkpoint files impose heavy burden to storage. Consider using cheaper HDD storage for checkpoining.

```bash
# make a new directory
mkdir ckpt-wiki-koen
mkdir ckpt-wiki-enko

# or link somewhere else, a heterogeneous storage
ln -s /path/to/one/storage ckpt-wiki-koen
ln -s /path/to/another/storage ckpt-wiki-enko

# likewise, you need to make following directories
mkdir -p dataset/tokenizer-corpus/tokenized-corpus
mkdir -p dataset/snutok-processed-archive
mkdir -p dataset/wiki-0420
mkdir -p dataset/modoo-translation
```

## Prepare corpus

### Tokenizer corpus

We sample 1M sentences from wikipedia dump to build our tokenizer corpus.

```bash
cp /data/s0/snutok/wiki-sentences/kowiki-0420-1M.txt data/tokenizer-corpus
cp /data/s0/snutok/wiki-sentences/enwiki-0420-1M.txt data/tokenizer-corpus
```

### Translation model corpus

We utilize parallel corpus. Detail will be documented soon.

```bash
cp /data/s0/snutok/translator-corpus/*.txt dataset/modoo-translation/
```

>

## Train tokenizer

+ [Reference](cho-scripts/train-wiki-tokenizer.sh)

## Tokenize corpus

+ [Reference](generate_tokenized_sentences.py)

## Preprocess corpus

+ [Reference ko-en](cho-scripts/preprocess-wiki-koen.sh)
+ [Reference en-ko](cho-scripts/preprocess-wiki-enko.sh)

## Train models

+ [Reference ko-en](cho-scripts/train-wiki-koen.sh)
+ [Reference en-ko](cho-scripts/train-wiki-enko.sh)

## Run Task

+ [Reference ko-en](cho-scripts/task-wiki-koen.sh)
+ [Reference en-ko](cho-scripts/task-wiki-enko.sh)

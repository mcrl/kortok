#!/bin/bash

dict="/home/n5/chanwoo/utils/mecab-ko/lib/mecab/dic/mecab-ko-dic"
ko_corpus="dataset/modoo-translation/ko_sentences.txt"
en_corpus="dataset/modoo-translation/en_sentences.txt"
merged_corpus="dataset/modoo-translation/merged_sentences.txt"

# train word tokenizers
python scripts/build_word_vocab.py \
    --vocab=64000 \
    --input_corpus $ko_corpus &

# generate mecab tokenized corpus
python scripts/mecab_tokenization.py \
    --input_corpus $ko_corpus \
    --output_dir dataset/modoo-translation/mecab_tokenized \
    --dicdir $dict

# train jamo tokenizers
python scripts/build_jamo_vocab.py \
    --vocab=200 \
    --input_corpus $ko_corpus &

# train char tokenizers
python scripts/build_char_vocab.py \
    --vocab=2000 \
    --input_corpus $ko_corpus &

wait

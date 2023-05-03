#!/bin/bash

# generate mecab tokenized corpus
python scripts/mecab_tokenization.py \
    --input_corpus dataset/modoo-translation/ko_sentences.txt \
    --output_dir dataset/modoo-translation/mecab_tokenized \
    --dicdir /home/n5/chanwoo/utils/mecab-ko/lib/mecab/dic/mecab-ko-dic

# train char tokenizers

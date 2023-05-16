#!/bin/bash

dict="/home/n5/chanwoo/utils/mecab-ko/lib/mecab/dic/mecab-ko-dic"
ko_corpus="wiki-corpus/kowiki-0420.txt"
en_corpus="wiki-corpus/enwiki-0420.txt"
mecab_ko_corpus_dir="wiki-corpus/mecab"
mecab_ko_corpus=$mecab_ko_corpus_dir"/kowiki-0420.txt"

tokenizer_dir="wiki-tokenizers"

rm -rf $mecab_ko_corpus_dir
mkdir -p $mecab_ko_corpus_dir
rm -rf $tokenizer_dir
mkdir -p $tokenizer_dir

# train word tokenizers
python scripts/build_word_vocab.py \
    --vocab=64000 \
    --output_dir $tokenizer_dir \
    --input_corpus $ko_corpus &

# generate mecab tokenized corpus
python scripts/mecab_tokenization.py \
    --input_corpus $ko_corpus \
    --output_dir $mecab_ko_corpus_dir \
    --dicdir $dict &

wait 

# train jamo tokenizers
python scripts/build_jamo_vocab.py \
    --vocab=200 \
    --output_dir $tokenizer_dir \
    --input_corpus $ko_corpus &

# train char tokenizers
python scripts/build_char_vocab.py \
    --vocab=2000 \
    --output_dir $tokenizer_dir \
    --input_corpus $ko_corpus &

wait

# train mecab tokenizers
python scripts/build_mecab_vocab.py \
    --vocab_size 8000 \
    --input_corpus $ko_corpus \
    --output_dir $tokenizer_dir \
    --dicdir $dict &

python scripts/build_mecab_vocab.py \
    --vocab_size 16000 \
    --input_corpus $ko_corpus \
    --output_dir $tokenizer_dir \
    --dicdir $dict &

wait

python scripts/build_mecab_vocab.py \
    --vocab_size 32000 \
    --input_corpus $ko_corpus \
    --output_dir $tokenizer_dir \
    --dicdir $dict &

python scripts/build_mecab_vocab.py \
    --vocab_size 64000 \
    --input_corpus $ko_corpus \
    --output_dir $tokenizer_dir \
    --dicdir $dict &

wait

number_array=(16000 32000 64000)
for number in "${number_array[@]}"
do
    python scripts/train_sentencepiece.py \
    --vocab_size $number \
    --tokenizer_type "ko" \
    --output_dir $tokenizer_dir \
    --input_ko_corpus $ko_corpus &
done
wait

number_array=(32000 64000)
for number in "${number_array[@]}"
do
    python scripts/train_sentencepiece.py \
        --vocab_size $number \
        --tokenizer_type="en" \
        --normalization_rule_name="nmt_nfkc" \
        --output_dir $tokenizer_dir \
        --input_en_corpus $en_corpus &
done
wait

number_array=(16000 32000 64000)
for number in "${number_array[@]}"
do
    python scripts/train_sentencepiece.py \
        --vocab_size $number \
        --tokenizer_type="mecab_tokenized" \
        --output_dir $tokenizer_dir \
        --input_mecab_corpus $mecab_ko_corpus &
done
wait

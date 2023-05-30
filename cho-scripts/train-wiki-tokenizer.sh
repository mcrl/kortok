#!/bin/bash

dict="/home/n5/chanwoo/utils/mecab-ko/lib/mecab/dic/mecab-ko-dic"
ko_corpus="dataset/tokenizer-corpus/kowiki-0420-1M.txt"
en_corpus="dataset/tokenizer-corpus/enwiki-0420-1M.txt"
mecab_ko_corpus="dataset/tokenizer-corpus/mecab_tokenized/kowiki-0420-1M.txt"
mecab_ko_corpus_dir="dataset/tokenizer-corpus/mecab_tokenized"
mecab_ko_corpus=$mecab_ko_corpus_dir"/kowiki-0420-1M.txt"

tokenizer_dir="wiki-tokenizers"

rm -rf $mecab_ko_corpus_dir
mkdir -p $mecab_ko_corpus_dir
rm -rf $tokenizer_dir
mkdir -p $tokenizer_dir


post-slack "train-wiki-tokenizer.sh started"

# generate mecab tokenized corpus
python scripts/mecab_tokenization.py \
    --input_corpus $ko_corpus \
    --output_dir $mecab_ko_corpus_dir \
    --dicdir $dict &
wait 

if [[ $? -ne 0 ]]; then
    post-slack "mecab_tokenization.py failed"
    exit 1
fi

# train word tokenizers
python scripts/build_word_vocab.py \
    --vocab=64000 \
    --output_dir $tokenizer_dir \
    --input_corpus $ko_corpus &


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

post-slack "build_word_vocab.py, build_jamo_vocab.py, build_char_vocab.py finished"

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

post-slack "build_mecab_vocab.py finished"

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

post-slack "build_mecab_vocab.py finished"

number_array=(32000 64000)
for number in "${number_array[@]}"
do
    python scripts/train_sentencepiece.py \
    --vocab_size $number \
    --tokenizer_type "ko" \
    --output_dir $tokenizer_dir \
    --input_ko_corpus $ko_corpus &
done
wait

post-slack "train_sentencepiece.py finished"

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

post-slack "train_sentencepiece.py finished"

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

post-slack "train_sentencepiece.py finished"
post-slack "train-wiki-tokenizer.sh finished"

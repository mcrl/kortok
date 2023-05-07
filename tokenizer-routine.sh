#!/bin/bash

dict="/home/n5/chanwoo/utils/mecab-ko/lib/mecab/dic/mecab-ko-dic"
ko_corpus="dataset/modoo-translation/ko_sentences.txt"
en_corpus="dataset/modoo-translation/en_sentences.txt"
merged_corpus="dataset/modoo-translation/merged_sentences.txt"
mecab_ko_corpus="dataset/modoo-translation/mecab_tokenized/ko_sentences.txt"

# train word tokenizers
python scripts/build_word_vocab.py \
    --vocab=64000 \
    --input_corpus $ko_corpus &

# train jamo tokenizers
python scripts/build_jamo_vocab.py \
    --vocab=200 \
    --input_corpus $ko_corpus &

# train char tokenizers
python scripts/build_char_vocab.py \
    --vocab=2000 \
    --input_corpus $ko_corpus &

# train mecab tokenizers
python scripts/build_mecab_vocab.py \
    --vocab_size 8000 \
    --input_corpus $ko_corpus \
    --dicdir $dict &

python scripts/build_mecab_vocab.py \
    --vocab_size 16000 \
    --input_corpus $ko_corpus \
    --dicdir $dict &

python scripts/build_mecab_vocab.py \
    --vocab_size 32000 \
    --input_corpus $ko_corpus \
    --dicdir $dict &

python scripts/build_mecab_vocab.py \
    --vocab_size 64000 \
    --input_corpus $ko_corpus \
    --dicdir $dict &

number_array=(4000 8000 16000 32000 64000)
for number in "${number_array[@]}"
do

    python scripts/train_sentencepiece.py \
    --vocab_size $number \
    --tokenizer_type "ko" \
    --input_ko_corpus $ko_corpus &


    python scripts/train_sentencepiece.py \
        --vocab_size $number \
        --tokenizer_type="en" \
        --normalization_rule_name="nmt_nfkc" \
        --input_en_corpus $en_corpus &
done

# generate mecab tokenized corpus
python scripts/mecab_tokenization.py \
    --input_corpus $ko_corpus \
    --output_dir dataset/modoo-translation/mecab_tokenized \
    --dicdir $dict &

wait

number_array=(4000 8000 16000 32000 64000)
for number in "${number_array[@]}"
do
    python scripts/train_sentencepiece.py \
        --vocab_size $number \
        --tokenizer_type="mecab_tokenized" \
        --input_mecab_corpus $mecab_ko_corpus &
done

wait

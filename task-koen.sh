#!/bin/bash

en_dic=en_sp-64k

function task(){
    ko_dic=$1
    device=$2
    ckpt=translation_ckpt/${ko_dic}-${en_dic}/ko-en/checkpoint_last.pt
    CUDA_VISIBLE_DEVICES=$device \
    fairseq-generate dataset/modoo-translation/${ko_dic}-${en_dic}/preprocessed/ko-en \
        --path $ckpt \
        --batch-size 512 \
        --beam 5 \
        --remove-bpe sentencepiece \
        > task-log/${ko_dic}-${en_dic}-generate.log
}

task char-2k 0 &
task jamo-200 1 &
task word-64k 2 &
task mecab_sp-4k 3 &
wait

task mecab_sp-8k 0 &
task mecab_sp-32k 1 &
task mecab_sp-64k 2 &
task mecab-8k 3 &
wait

task mecab-16k 0 &
task mecab-32k 1 &
task mecab-64k 2 &
task sp-4k 3 &
wait

task sp-8k 0 &
task sp-32k 2 &
task sp-64k 3 &
wait
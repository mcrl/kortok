#!/bin/bash

en_dic=en_sp-64k
src_lang=ko
dest_lang=en
ckpt_dir=ckpt-wiki-koen
data_dir=wiki-0420

function task(){
    ko_dic=$1
    device=$2

    left_dic=$ko_dic
    right_dic=$en_dic
    ckpt=${ckpt_dir}/${left_dic}-${right_dic}/${src_lang}-${dest_lang}/checkpoint_last.pt
    CUDA_VISIBLE_DEVICES=$device \
    fairseq-generate dataset/${data_dir}/${left_dic}-${right_dic}/preprocessed/${src_lang}-${dest_lang} \
        --path $ckpt \
        --batch-size 512 \
        --beam 5 \
        --remove-bpe sentencepiece \
        > task-log/${data_dir}/${left_dic}-${right_dic}-generate.log
}

mkdir -p task-log/${data_dir}

task char-2k 0 &
task jamo-200 1 &
task word-64k 2 &
task sp-64k 3 &
wait

task mecab_sp-32k 0 &
task mecab_sp-64k 1 &
task mecab-8k 2 &
task mecab-16k 3 &
wait

task mecab-32k 0 &
task mecab-64k 1 &
task sp-32k 3 &
wait

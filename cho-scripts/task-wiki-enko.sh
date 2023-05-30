#!/bin/bash

en_dic=en_sp-64k
src_lang=en
dest_lang=ko
ckpt_dir=ckpt-wiki-enko
data_dir=snutok-processed-archive

function task(){
    ko_dic=$1
    device=$2

    left_dic=$en_dic
    right_dic=$ko_dic
    ckpt=${ckpt_dir}/${left_dic}-${right_dic}/${src_lang}-${dest_lang}/checkpoint_last.pt

    post-slack "Task $1 for en-ko task started"

    CUDA_VISIBLE_DEVICES=$device \
    fairseq-generate dataset/${data_dir}/${left_dic}-${right_dic}/preprocessed/${src_lang}-${dest_lang} \
        --path $ckpt \
        --batch-size 512 \
        --beam 5 \
        --remove-bpe sentencepiece \
        > task-log/${data_dir}/${left_dic}-${right_dic}-generate.log

    if [ $? -ne 0 ]; then
        post-slack "##############################"
        post-slack "Task $1 for en-ko task FAILED"
        post-slack "##############################"
        exit 1
    else 
        post-slack "Task $1 for en-ko task done"
    fi
}

mkdir -p task-log/${data_dir}

task mecab-32k 0 &
task mecab-64k 1 &
task sp-32k 2 &
task sp-64k 3 &
wait

task mecab_sp-32k 0 &
task mecab_sp-64k 1 &
task mecab-8k 2 &
task mecab-16k 3 &
wait

task char-2k 0 &
task jamo-200 1 &
task word-64k 2 &
wait

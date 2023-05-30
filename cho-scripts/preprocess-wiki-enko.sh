#!/bin/bash

workers=16
srclang=en
destlang=ko

data_src=wiki-0420
resourcedir=wiki-tokenizers
dest_path=snutok-preprocessed-archive
logdir=logs/preprocess-log

function submit {
    source_dict=$1
    target_dict=$2
    job_name="${source_dict}-${target_dict}"
    datadir=$PWD/dataset/$data_src/$job_name
    srclang_srcdir=$PWD/dataset/$data_src/tokenized/$source_dict
    destlang_srcdir=$PWD/dataset/$data_src/tokenized/$target_dict
    dest=$PWD/dataset/$dest_path/$job_name/preprocessed/$srclang-$destlang

    rm -rf $dest/*
    rm -rf $datadir
    mkdir -p $datadir

    ln -s $srclang_srcdir/train.$srclang $datadir/train.$srclang
    ln -s $srclang_srcdir/validation.$srclang $datadir/validation.$srclang
    ln -s $srclang_srcdir/test.$srclang $datadir/test.$srclang
    ln -s $destlang_srcdir/train.$destlang $datadir/train.$destlang
    ln -s $destlang_srcdir/validation.$destlang $datadir/validation.$destlang
    ln -s $destlang_srcdir/test.$destlang $datadir/test.$destlang

    echo "Submitted $job_name"
    post-slack "preprocess-wiki-enko.sh ${job_name} started"
    
    fairseq-preprocess \
    --source-lang $srclang \
    --target-lang $destlang \
    --trainpref $datadir/train \
    --validpref $datadir/validation \
    --testpref $datadir/test \
    --destdir $dest \
    --srcdict $resourcedir/$source_dict/fairseq.vocab \
    --tgtdict $resourcedir/$target_dict/fairseq.vocab \
    --workers $workers \
    1> $logdir/$job_name.log 2> $logdir/$job_name.err

    if [ $? -ne 0 ]; then
        post-slack "preprocess-wiki-enko.sh ${job_name} failed"
        exit 1
    else
        post-slack "preprocess-wiki-enko.sh ${job_name} done"
    fi
}

post-slack preprocess-wiki-enko.sh started

submit en_sp-64k char-2k
submit en_sp-64k jamo-200
submit en_sp-64k word-64k
submit en_sp-64k mecab_sp-32k
submit en_sp-64k mecab_sp-64k
submit en_sp-64k mecab-8k
submit en_sp-64k mecab-16k
submit en_sp-64k mecab-32k
submit en_sp-64k mecab-64k
submit en_sp-64k sp-32k
submit en_sp-64k sp-64k

wait

post-slack preprocess-wiki-enko.sh done
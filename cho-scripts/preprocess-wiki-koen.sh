#!/bin/bash

workers=16
srclang=ko
destlang=en

data_src=wiki-0420
dest_path=snutok-processed-archive
resourcedir=wiki-tokenizers
logdir=logs/preprocess-log

function submit {
    source_dict=$1
    target_dict=$2
    job_name="${source_dict}-${target_dict}"
    datadir=$PWD/dataset/$data_src/$job_name
    srclang_srcdir=$PWD/dataset/$data_src/tokenized/$source_dict
    destlang_srcdir=$PWD/dataset/$data_src/tokenized/$target_dict
    dest=$PWD/dataset/$dest_path/$job_name/preprocessed/$srclang-$destlang

    rm -rf $dest
    rm -rf $datadir
    mkdir -p $datadir
    mkdir -p $dest

    ln -s $srclang_srcdir/train.$srclang $datadir/train.$srclang
    ln -s $srclang_srcdir/validation.$srclang $datadir/validation.$srclang
    ln -s $srclang_srcdir/test.$srclang $datadir/test.$srclang
    ln -s $destlang_srcdir/train.$destlang $datadir/train.$destlang
    ln -s $destlang_srcdir/validation.$destlang $datadir/validation.$destlang
    ln -s $destlang_srcdir/test.$destlang $datadir/test.$destlang

    echo "Submitted $job_name"
    post-slack "preprocess-wiki-koen.sh ${job_name} started"

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
        post-slack "preprocess-wiki-koen.sh ${job_name} failed"
        exit 1
    else
        post-slack "preprocess-wiki-koen.sh ${job_name} done"
    fi
}

post-slack 'preprocess-wiki-koen.sh started'

submit char-2k en_sp-64k
submit jamo-200 en_sp-64k
submit word-64k en_sp-64k
submit mecab_sp-32k en_sp-64k
submit mecab_sp-64k en_sp-64k
submit mecab-8k en_sp-64k
submit mecab-16k en_sp-64k
submit mecab-32k en_sp-64k
submit mecab-64k en_sp-64k
submit sp-32k en_sp-64k
submit sp-64k en_sp-64k

wait

post-slack 'preprocess-wiki-koen.sh done'

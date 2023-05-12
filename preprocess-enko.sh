#!/bin/bash

workers=4
srclang=en
destlang=ko

data_src=modoo-translation
resourcedir=resources

function submit {
    source_dict=$1
    target_dict=$2
    job_name="${source_dict}-${target_dict}"
    datadir=$PWD/dataset/$data_src/$job_name
    srclang_srcdir=$PWD/dataset/$data_src/tokenized/$source_dict
    destlang_srcdir=$PWD/dataset/$data_src/tokenized/$target_dict
    dest=$PWD/dataset/$data_src/$job_name/preprocessed/$srclang-$destlang

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
    fairseq-preprocess \
    --source-lang $srclang \
    --target-lang $destlang \
    --trainpref $datadir/train \
    --validpref $datadir/validation \
    --testpref $datadir/test \
    --destdir $dest \
    --srcdict $resourcedir/$source_dict/fairseq.vocab \
    --tgtdict $resourcedir/$target_dict/fairseq.vocab \
    --workers $workers
}

submit en_sp-64k char-2k
submit en_sp-64k jamo-200
submit en_sp-64k word-64k
submit en_sp-64k mecab_sp-4k
submit en_sp-64k mecab_sp-8k
submit en_sp-64k mecab_sp-16k
submit en_sp-64k mecab_sp-32k
submit en_sp-64k mecab_sp-64k
submit en_sp-64k mecab-8k
submit en_sp-64k mecab-16k
submit en_sp-64k mecab-32k
submit en_sp-64k mecab-64k
submit en_sp-64k sp-4k
submit en_sp-64k sp-8k
submit en_sp-64k sp-16k
submit en_sp-64k sp-32k
submit en_sp-64k sp-64k

submit en_sp-32k char-2k
submit en_sp-32k jamo-200
submit en_sp-32k word-64k
submit en_sp-32k mecab_sp-4k
submit en_sp-32k mecab_sp-8k
submit en_sp-32k mecab_sp-16k
submit en_sp-32k mecab_sp-32k
submit en_sp-32k mecab_sp-64k
submit en_sp-32k mecab-8k
submit en_sp-32k mecab-16k
submit en_sp-32k mecab-32k
submit en_sp-32k mecab-64k
submit en_sp-32k sp-4k
submit en_sp-32k sp-8k
submit en_sp-32k sp-16k
submit en_sp-32k sp-32k
submit en_sp-32k sp-64k

wait

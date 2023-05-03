#!/bin/bash

workers=4
srclang=ko
destlang=en

datadir=dataset/modoo-translation/processed
resourcedir=resources

function submit {
    source_dict=$1
    target_dict=$2
    job_name="${source_dict}-${target_dict}"
    dest=dataset/modoo-translation/$job_name/preprocessed/$srclang-$destlang
    rm -rf $dest
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

submit char-2k en_sp-32k
submit char-2k en_sp-64k
submit sp-4k en_sp-32k
submit sp-8k en_sp-32k
submit sp-16k en_sp-32k
submit sp-32k en_sp-32k
submit sp-64k en_sp-32k
submit sp-4k en_sp-64k
submit sp-8k en_sp-64k
submit sp-16k en_sp-64k
submit sp-32k en_sp-64k
submit sp-64k en_sp-64k

wait
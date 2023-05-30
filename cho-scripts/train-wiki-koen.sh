en_dic=en_sp-64k
datasrc=snutok-processed-archive

function train(){
    post-slack "Training $1 for ko-en task"
    ko_dic=$1
    fairseq-train dataset/$datasrc/${ko_dic}-${en_dic}/preprocessed/ko-en \
    --arch transformer \
    --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-epoch 50 \
    --batch-size 128 \
    --save-dir ckpt-wiki-koen/${ko_dic}-${en_dic}/ko-en \
    --disable-validation \
    --memory-efficient-fp16 \
    --save-interval 5

    if [ $? -ne 0 ]; then
        post-slack "Training $1 for ko-en task failed"
        exit 1
    else 
        post-slack "Training $1 for ko-en task done"
    fi
}

train char-2k
train jamo-200
train word-64k
train mecab_sp-32k
train mecab_sp-64k
train mecab-8k
train mecab-16k
train mecab-32k
train mecab-64k
train sp-32k
train sp-64k
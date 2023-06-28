#!/bin/bash

echo "Prepare KorQuAD dataset and Model"
# python prepare_korquad.py

echo "Run KorQuAD training"
PYTHONPATH="./$PYTHONPATH" python tasks/korquad/run_train.py --tokenizer klue
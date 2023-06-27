#!/bin/bash

echo "Downloading dataset and pretrained parameters"
python download.py

echo "Fine-tuning the model"
PYTHONPATH=".:$PYTHONPATH" python tasks/korsts/run_train.py --tokenizer klue

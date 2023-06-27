import argparse
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import BertConfig


from tokenizer import (
    CharTokenizer,
    JamoTokenizer,
    MeCabSentencePieceTokenizer,
    MeCabTokenizer,
    SentencePieceTokenizer,
    Vocab,
    WordTokenizer,
)

from pretrain_gpt.logger import get_logger


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

TOK_INFO_JSON = "build_info.json"
VOCAB_FILE="tok.vocab"
MODEL_INFO_TXT = "tok.model"
CKPT_DIR = "checkpoints"
LOG_DIR = "gpt-logs"
MECAB_DICT = "/home/n5/chanwoo/utils/mecab-ko/lib/mecab/dic/mecab-ko-dic"
DEFAULT_RESOURCE_DIR = "wiki-tokenizers"

def main(args):
    # config
    set_seed(42)

    experiment_name = args.name if args.name else args.tokenizer_name

    checkpoint_dir = os.path.join(CKPT_DIR, experiment_name)
    log_dir = os.path.join(LOG_DIR, experiment_name)
    
    checkpoint_dir_path = Path(checkpoint_dir)
    if not checkpoint_dir_path.exists():
        checkpoint_dir_path.mkdir(parents=True)
    log_dir_path = Path(log_dir)
    if not log_dir_path.exists():
        log_dir_path.mkdir(parents=True)
    
    logger = get_logger(log_path=os.path.join(log_dir, "logs.txt"))
    logger.info(f"Setting up {experiment_name} experiment...")
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    logger.info(f"Log directory: {log_dir}")

    # 기본적인 모듈들 생성 (vocab, tokenizer)
    tokenizer_name = args.tokenizer_name
    tokenizer_dir = os.path.join(args.resource_dir, tokenizer_name)
    logger.info(f"get vocab and tokenizer from {tokenizer_dir}")
    vocab = Vocab(os.path.join(tokenizer_dir, VOCAB_FILE))
    if tokenizer_name.startswith("mecab-"):
        tokenizer = MeCabTokenizer(os.path.join(tokenizer_dir, TOK_INFO_JSON),mecab_path=MECAB_DICT)
    elif tokenizer_name.startswith("sp-"):
        tokenizer = SentencePieceTokenizer(os.path.join(tokenizer_dir, MODEL_INFO_TXT))
    elif tokenizer_name.startswith("mecab_sp-"):
        mecab = MeCabTokenizer(os.path.join(tokenizer_dir, TOK_INFO_JSON), mecab_path=MECAB_DICT)
        sp = SentencePieceTokenizer(os.path.join(tokenizer_dir, MODEL_INFO_TXT))
        tokenizer = MeCabSentencePieceTokenizer(mecab, sp)
    elif tokenizer_name.startswith("char-"):
        tokenizer = CharTokenizer()
    elif tokenizer_name.startswith("word-"):
        tokenizer = WordTokenizer()
    elif tokenizer_name.startswith("jamo-"):
        tokenizer = JamoTokenizer()
    else:
        raise ValueError("Wrong tokenizer name.")

    exit(0)
    # 모델에 넣을 데이터 준비
    # label-to-index
    label_to_index = {"0": 0, "1": 1}
    # Train
    logger.info(f"read training data from {config.train_path}")
    train_sentence_as, train_sentence_bs, train_labels = load_data(config.train_path, label_to_index)
    # Dev
    logger.info(f"read dev data from {config.dev_path}")
    dev_sentence_as, dev_sentence_bs, dev_labels = load_data(config.dev_path, label_to_index)
    # Test
    logger.info(f"read test data from {config.test_path}")
    test_sentence_as, test_sentence_bs, test_labels = load_data(config.test_path, label_to_index)

    # 데이터로 dataloader 만들기
    # Train
    logger.info("create data loader using training data")
    train_dataset = PAWSDataset(
        train_sentence_as, train_sentence_bs, train_labels, vocab, tokenizer, config.max_sequence_length
    )
    train_random_sampler = RandomSampler(train_dataset)
    train_data_loader = DataLoader(train_dataset, sampler=train_random_sampler, batch_size=config.batch_size)
    # Dev
    logger.info("create data loader using dev data")
    dev_dataset = PAWSDataset(
        dev_sentence_as, dev_sentence_bs, dev_labels, vocab, tokenizer, config.max_sequence_length
    )
    dev_data_loader = DataLoader(dev_dataset, batch_size=1024)
    # Test
    logger.info("create data loader using test data")
    test_dataset = PAWSDataset(
        test_sentence_as, test_sentence_bs, test_labels, vocab, tokenizer, config.max_sequence_length
    )
    test_data_loader = DataLoader(test_dataset, batch_size=1024)

    # Summary Writer 준비
    summary_writer = SummaryWriter(log_dir=config.summary_dir)

    # 모델을 준비하는 코드
    logger.info("initialize model and convert bert pretrained weight")
    bert_config = BertConfig.from_json_file(
        os.path.join(config.resource_dir, config.tokenizer, config.bert_config_file_name)
    )
    model = PAWSModel(bert_config, config.dropout_prob)
    model.bert = load_pretrained_bert(
        bert_config, os.path.join(config.resource_dir, config.tokenizer, config.pretrained_bert_file_name)
    )

    trainer = Trainer(config, model, train_data_loader, dev_data_loader, test_data_loader, logger, summary_writer)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    experiment_group = parser.add_argument_group("experiment")
    experiment_group.add_argument("--name", type=str)
    experiment_group.add_argument("--tokenizer-name", type=str, required=True)
    experiment_group.add_argument("--resource-dir", type=str, default=DEFAULT_RESOURCE_DIR)

    args = parser.parse_args() 
    main(args)

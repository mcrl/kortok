from tokenizer import *
from pathlib import Path
from tqdm import tqdm
from functools import partial
import os
import sys

DICT_PATH = "/home/n5/chanwoo/utils/mecab-ko/lib/mecab/dic/mecab-ko-dic"


def tokenize_sentences(input_file, output_file, tokenizer: BaseTokenizer, pbar=False):
    input_file = Path(input_file)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(input_file, "r") as f:
        lines = f.readlines()

    print("Tokenizing...")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")

    with open(output_file, "w") as f:
        for line in tqdm(lines) if pbar else lines:
            tokenized = tokenizer.tokenize(line)
            tokenized = " ".join(tokenized)
            f.write(tokenized + "\n")
        print("Saving File...... ", end="")
        sys.stdout.flush()

    print("Done!")
    print("=" * 30)


if __name__ == "__main__":
    input_file = "dataset/modoo-translation/ko_sentences.txt"
    resources = os.listdir("resources")

    tasks = []
    # create char tokenizer task
    output_file = "dataset/modoo-translation/tokenized/ko_sentences_char-2k.txt"
    tokenizer = CharTokenizer()
    task = partial(tokenize_sentences, input_file, output_file, tokenizer)
    print(f"char_tokenizer")
    tasks.append(task)

    # JamoTokenizer
    output_file = "dataset/modoo-translation/tokenized/ko_sentences_jamo-200.txt"
    tokenizer = JamoTokenizer()
    task = partial(tokenize_sentences, input_file, output_file, tokenizer)
    print(f"jamo_tokenizer")
    tasks.append(task)

    # WordTokenizer
    output_file = "dataset/modoo-translation/tokenized/ko_sentences_word-64k.txt"
    tokenizer = WordTokenizer()
    task = partial(tokenize_sentences, input_file, output_file, tokenizer)
    print(f"word_tokenizer")
    tasks.append(task)

    # MecabTokenizer
    mecab_resources = [r for r in resources if r.startswith("mecab-")]
    for mecab_resource in mecab_resources:
        output_file = f"dataset/modoo-translation/tokenized/ko_sentences_{mecab_resource}.txt"
        tokenizer = MeCabTokenizer(mecab_path=DICT_PATH, config_path=f"resources/{mecab_resource}/tok.json")
        task = partial(tokenize_sentences, input_file, output_file, tokenizer)
        print(f"mecab_resource: {mecab_resource}")
        tasks.append(task)

    # SentencePieceTokenizer
    sp_resources = [r for r in resources if r.startswith("sp-")]
    for sp_resource in sp_resources:
        output_file = f"dataset/modoo-translation/tokenized/ko_sentences_{sp_resource}.txt"
        tokenizer = SentencePieceTokenizer(model_path=f"resources/{sp_resource}/tok.model")
        task = partial(tokenize_sentences, input_file, output_file, tokenizer)
        print(f"sp_resource: {sp_resource}")
        tasks.append(task)

    # en_sp tokenizer
    input_file = "dataset/modoo-translation/en_sentences.txt"
    en_sp_resources = [r for r in resources if r.startswith("en_sp-")]
    for en_sp_resource in en_sp_resources:
        output_file = f"dataset/modoo-translation/tokenized/en_sentences_{en_sp_resource}.txt"
        tokenizer = SentencePieceTokenizer(model_path=f"resources/{en_sp_resource}/tok.model")
        task = partial(tokenize_sentences, input_file, output_file, tokenizer)
        tasks.append(task)
        print(f"en_sp_resource: {en_sp_resource}")

    # MeCabSentencePieceTokenizer
    input_file = "dataset/modoo-translation/ko_sentences.txt"
    mecab_sp_resources = [r for r in resources if r.startswith("mecab_sp-")]
    for mecab_sp_resource in mecab_sp_resources:
        output_file = f"dataset/modoo-translation/tokenized/ko_sentences_{mecab_sp_resource}.txt"
        tokenizer = MeCabSentencePieceTokenizer(
            mecab=MeCabTokenizer(mecab_path=DICT_PATH, config_path="resources/mecab-16k/tok.json"),
            sp=SentencePieceTokenizer(model_path=f"resources/{mecab_sp_resource}/tok.model"),
        )
        task = partial(tokenize_sentences, input_file, output_file, tokenizer)
        print(f"mecab_sp_resource: {mecab_sp_resource}")
        tasks.append(task)

    assert len(tasks) == len(resources)

    print("=" * 30)
    print(f"Total {len(tasks)} tasks")
    print("=" * 30)
    # run all tasks
    for task in tasks:
        task(pbar=True)

    print("Done!")

from tokenizer import *
from pathlib import Path
from tqdm import tqdm
from functools import partial
import os
from multiprocessing import Process
from argparse import ArgumentParser

DICT_PATH = "/home/n5/chanwoo/utils/mecab-ko/lib/mecab/dic/mecab-ko-dic"
KO_CORPUSES = [
    Path("dataset/modoo-translation/processed/train.ko"),
    Path("dataset/modoo-translation/processed/validation.ko"),
    Path("dataset/modoo-translation/processed/test.ko"),
]
EN_CORPUSES = [
    Path("dataset/modoo-translation/processed/train.en"),
    Path("dataset/modoo-translation/processed/validation.en"),
    Path("dataset/modoo-translation/processed/test.en"),
]
OUTPUT_PATH = "dataset/wiki-0420/tokenized"
RESOURCES = "resources"


def tokenize_sentences(input_file, output_file, tokenizer: BaseTokenizer, pbar=False):
    input_file = Path(input_file)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(input_file, "r") as f:
        lines = f.readlines()

    with open(output_file, "w") as f:
        for line in tqdm(lines) if pbar else lines:
            tokenized = tokenizer.tokenize(line)
            tokenized = " ".join(tokenized)
            f.write(tokenized + "\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--resources", type=str, default=RESOURCES)
    parser.add_argument("--output", type=str, default=OUTPUT_PATH)
    args = parser.parse_args()
    processes = []
    resources = os.listdir(args.resources)
    output_path = Path(args.output)
    os.makedirs(output_path, exist_ok=True)

    tokenizer = CharTokenizer()
    for input_file in KO_CORPUSES:
        output_file = f"{output_path}/char-2k/{input_file.name}"
        task = partial(tokenize_sentences, input_file, output_file, tokenizer)
        process = Process(target=task)
        process.start()
        processes.append(process)
        print(f"CharTokenizer: {input_file.name} is processing...")

    # JamoTokenizer
    tokenizer = JamoTokenizer()
    for input_file in KO_CORPUSES:
        output_file = f"{output_path}/jamo-200/{input_file.name}"
        task = partial(tokenize_sentences, input_file, output_file, tokenizer)
        process = Process(target=task)
        process.start()
        processes.append(process)
        print(f"JamoTokenizer: {input_file.name} is processing...")

    # MecabTokenizer
    mecab_resources = [r for r in resources if r.startswith("mecab-")]
    for mecab_resource in mecab_resources:
        tokenizer = MeCabTokenizer(mecab_path=DICT_PATH, config_path=f"resources/{mecab_resource}/tok.json")
        for input_file in KO_CORPUSES:
            output_file = f"{output_path}/{mecab_resource}/{input_file.name}"
            task = partial(tokenize_sentences, input_file, output_file, tokenizer)
            process = Process(target=task)
            process.start()
            processes.append(process)
            print(f"MeCabTokenizer: {input_file.name} is processing with {mecab_resource}...")

    # SentencePieceTokenizer
    sp_resources = [r for r in resources if r.startswith("sp-")]
    for sp_resource in sp_resources:
        tokenizer = SentencePieceTokenizer(model_path=f"resources/{sp_resource}/tok.model")
        for input_file in KO_CORPUSES:
            output_file = f"{output_path}/{sp_resource}/{input_file.name}"
            task = partial(tokenize_sentences, input_file, output_file, tokenizer)
            process = Process(target=task)
            process.start()
            processes.append(process)
            print(f"SentencePieceTokenizer: {input_file.name} is processing with {sp_resource}...")

    # en_sp tokenizer
    en_sp_resources = [r for r in resources if r.startswith("en_sp-")]
    for en_sp_resource in en_sp_resources:
        tokenizer = SentencePieceTokenizer(model_path=f"resources/{en_sp_resource}/tok.model")
        for input_file in EN_CORPUSES:
            output_file = f"{output_path}/{en_sp_resource}/{input_file.name}"
            task = partial(tokenize_sentences, input_file, output_file, tokenizer)
            process = Process(target=task)
            process.start()
            processes.append(process)
            print(f"SentencePieceTokenizer: {input_file.name} is processing with {en_sp_resource}...")

    # MeCabSentencePieceTokenizer
    mecab_sp_resources = [r for r in resources if r.startswith("mecab_sp-")]
    for mecab_sp_resource in mecab_sp_resources:
        tokenizer = MeCabSentencePieceTokenizer(
            mecab=MeCabTokenizer(mecab_path=DICT_PATH, config_path="resources/mecab-16k/tok.json"),
            sp=SentencePieceTokenizer(model_path=f"resources/{mecab_sp_resource}/tok.model"),
        )
        for input_file in KO_CORPUSES:
            output_file = f"{output_path}/{mecab_sp_resource}/{input_file.name}"
            task = partial(tokenize_sentences, input_file, output_file, tokenizer)
            process = Process(target=task)
            process.start()
            processes.append(process)
            print(f"MeCabSentencePieceTokenizer: {input_file.name} is processing with {mecab_sp_resource}...")

    # special group: serial run
    print("=== Special group: serial run ===")
    # WordTokenizer
    print("WordTokenizer")
    tokenizer = WordTokenizer()
    for input_file in KO_CORPUSES:
        output_file = f"{output_path}/word-64k/{input_file.name}"
        task = partial(tokenize_sentences, input_file, output_file, tokenizer)
        task(pbar=True)

    print("=== Waiting for all processes to finish ===")
    for process in tqdm(processes):
        process.join()
    print("=== All processes finished ===")

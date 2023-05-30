import argparse
import json
import os
import time
from functools import partial
from multiprocessing import Pool
from typing import List
import multiprocessing as mp

import MeCab

INPUT_CORPUS = "dataset/modoo-translation/ko_sentences.txt"
OUTPUT_DIR = "./dataset/modoo-translation/mecab_tokenized"
DEFAULT_DICT = "/usr/local/lib/mecab/dic/mecab-ko-dic"
global_tokenizer = None  # type: Optional[MeCab.Tagger]


def tokenize(text: str, space_symbol: str = "▃") -> List[str]:
    # empty string or only spaces
    if not text or text.isspace():
        return []
    text = text.strip()
    text_ptr = 0
    tokenized = []
    for mor in global_tokenizer.parse(text).split("\n"):
        if "\t" in mor:
            splitted = mor.split("\t")
            token = splitted[0]
            # pos = splitted[1].split(",", 1)[0]

            if text[text_ptr] == " ":
                while text[text_ptr] == " ":
                    text_ptr += 1
                if text[text_ptr] != token[0]:
                    print(f"token: {token[0]}, text: {text[text_ptr]}")
                    assert text[text_ptr] == token[0]

                tokenized.append(space_symbol)

            tokenized.append(token)
            text_ptr += len(token)

    return tokenized


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--space_symbol", type=str, default="▃")
    parser.add_argument("--n_jobs", type=int, default=mp.cpu_count())
    parser.add_argument("--dicdir", type=str, default=DEFAULT_DICT, help="mecab dictionary directory")
    parser.add_argument("--input_corpus", type=str, default=INPUT_CORPUS, help="input corpus path")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="output directory")
    args = vars(parser.parse_args())
    print(args)

    os.makedirs(args["output_dir"], exist_ok=True)
    global_tokenizer = MeCab.Tagger(f"-r/dev/null --dicdir {args['dicdir']}")

    # set tokenizing func
    tokenize_fn = partial(tokenize, space_symbol=args["space_symbol"])

    start_time = time.time()
    print(f"start tokenization ...")
    with open(args["input_corpus"], "r", encoding="utf-8") as f:
        with Pool(args["n_jobs"]) as p:
            tokenized = p.map(tokenize_fn, f)
    elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f"complete tokenization for all files. (elapsed time: {elapsed_time})")

    # mecab tokenized corpus
    with open(os.path.join(args["output_dir"], os.path.basename(args["input_corpus"])), "w", encoding="utf-8") as f:
        for tokens in tokenized:
            f.write(" ".join(tokens) + "\n")

    # mecab config
    print("write mecab config file...")
    output_config_path = os.path.join(args["output_dir"], "tok.json")
    with open(output_config_path, "w", encoding="utf-8") as f:
        json.dump(args, f, indent=4)

    print("done.")

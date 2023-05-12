import argparse
import json
import os
import time
from collections import Counter
from itertools import chain
from multiprocessing import Pool
from typing import List
import multiprocessing as mp

INPUT_CORPUS = "dataset/modoo-translation/ko_sentences.txt"
OUTPUT_DIR = "./resources"


def tokenize(text: str) -> List[str]:
    text = text.strip().replace(" ", "▁")
    return list(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, required=True)

    parser.add_argument("--pad_piece", type=str, default="[PAD]", help="index=0")
    parser.add_argument("--unk_piece", type=str, default="[UNK]", help="index=1")
    parser.add_argument("--bos_piece", type=str, default="[BOS]", help="index=2")
    parser.add_argument("--eos_piece", type=str, default="[EOS]", help="index=3")
    parser.add_argument("--input_corpus", type=str, default=INPUT_CORPUS, help="input corpus path")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="output directory")
    parser.add_argument(
        "--special_symbols",
        type=str,
        default="[CLS],[SEP],[MASK]",
        help="Special tokens. You can pass a comma-separated list of special tokens.",
    )
    parser.add_argument("--n_jobs", type=int, default=mp.cpu_count())
    args = vars(parser.parse_args())
    print(args)

    # set output dir
    output_dir = os.path.join(args["output_dir"], f"char-{args['vocab_size']//1000}k")
    os.makedirs(output_dir, exist_ok=True)

    # save arguments info
    output_info_path = os.path.join(output_dir, "build_info.json")
    with open(output_info_path, "w", encoding="utf-8") as f:
        json.dump(args, f, indent=4)

    counter = Counter()
    start_time = time.time()
    print(f"start tokenization ...")
    with open(args["input_corpus"], "r", encoding="utf-8") as f:
        with Pool(args["n_jobs"]) as p:
            tokenized = p.map(tokenize, f)
            counter.update(chain.from_iterable(tokenized))
    elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f"complete tokenization for all files. (elapsed time: {elapsed_time})")

    # special tokens
    special_tokens = [args["pad_piece"], args["unk_piece"], args["bos_piece"], args["eos_piece"]]
    special_tokens.extend(args["special_symbols"].split(","))

    # slice with vocab size
    vocab = counter.most_common(args["vocab_size"] - len(special_tokens))

    # print out-of-vocabulary
    total_freq = sum(counter.values())
    oov_freq = total_freq - sum([v[1] for v in vocab])
    print(f"oov: {oov_freq}/{total_freq} ({oov_freq * 100.0 / total_freq:.2f}%)")

    # save char vocab
    output_vocab_path = os.path.join(output_dir, "tok.vocab")
    with open(output_vocab_path, "w", encoding="utf-8") as f:
        for token in special_tokens:
            f.write(f"{token}\t-1\n")
        for token, freq in vocab:
            f.write(f"{token}\t{freq}\n")

    # save fairseq vocab
    with open(os.path.join(output_dir, "fairseq.vocab"), "w") as fout:
        with open(os.path.join(output_dir, "tok.vocab"), "r") as fin:
            start_idx = 4 + len(args["special_symbols"].split(","))  # pad, unk, bos, eos + special_symbols
            for line in fin.readlines()[start_idx:]:
                splitted = line.split("\t")
                fout.write(f"{' '.join(splitted)}")

    print("done.")

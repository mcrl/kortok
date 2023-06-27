from transformers import AutoTokenizer, AutoModel
import shutil
import os
import requests


from tqdm import tqdm


def save_dataset(dataset, path):
    def _write_contents(fileio, dataset):
        for example in tqdm(dataset):
            guid = example["guid"]
            sentence1 = example["sentence1"]
            sentence2 = example["sentence2"]
            score = example["labels"]["label"]

            genre = "main-captions"
            filename = "klue-sts-v1-train"
            year = "2021"
            ident = guid.split("_")[-1]
            score = str(score)
            line = f"{genre}\t{filename}\t{year}\t{ident}\t{score}\t{sentence1}\t{sentence2}\n"
            fileio.write(line)

    with open(path, "w") as f:
        f.write("genre\tfilename\tyear\tid\tscore\tsentence1\tsentence2\n")
        _write_contents(f, dataset)


if False:
    # KLUE-STS 데이터셋을 다운로드 받아서 tsv 파일로 저장
    dataset = load_dataset("klue", "sts")

    dirpath = "dataset/nlu_tasks/korsts"
    train_file = os.path.join(dirpath, "train.tsv")
    dev_file = os.path.join(dirpath, "dev.tsv")
    test_file = os.path.join(dirpath, "test.tsv")

    tdataset = dataset["train"]
    save_dataset(tdataset, train_file)
    vdataset = dataset["validation"]
    save_dataset(vdataset, dev_file)
    save_dataset(vdataset, test_file)

if True:
    # KORSTS 데이터셋을 다운로드 받아서 tsv 파일로 저장

    dirpath = "dataset/nlu_tasks/korsts"
    train_file = os.path.join(dirpath, "sts-train.tsv")
    dev_file = os.path.join(dirpath, "sts-dev.tsv")
    test_file = os.path.join(dirpath, "sts-test.tsv")

    # download text file from https://raw.githubusercontent.com/kakaobrain/kor-nlu-datasets/master/KorSTS/sts-dev.tsv
    url = "https://raw.githubusercontent.com/kakaobrain/kor-nlu-datasets/master/KorSTS/sts-train.tsv"
    r = requests.get(url)
    with open(train_file, "w") as f:
        f.write(r.text)

    url = "https://raw.githubusercontent.com/kakaobrain/kor-nlu-datasets/master/KorSTS/sts-dev.tsv"
    r = requests.get(url)
    with open(dev_file, "w") as f:
        f.write(r.text)

    url = "https://raw.githubusercontent.com/kakaobrain/kor-nlu-datasets/master/KorSTS/sts-test.tsv"
    r = requests.get(url)
    with open(test_file, "w") as f:
        f.write(r.text)

target_dir = "resources/klue"
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
tokenizer.save_pretrained("klue.tok")
target_path = os.path.join(target_dir, "tok.vocab")
shutil.copy("klue.tok/vocab.txt", target_path)


model = AutoModel.from_pretrained("klue/bert-base")

# save bert config
save_dir = "klue.bert"
model.config.save_pretrained(save_dir)
# save bert model
model.save_pretrained(save_dir)


shutil.copy(save_dir + "/config.json", "resources/klue/bert_config.json")
shutil.copy(save_dir + "/pytorch_model.bin", "resources/klue/bert_model.pth")

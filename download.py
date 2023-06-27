from transformers import AutoTokenizer, AutoModel
from tokenizer.vocab import Vocab
import shutil


from datasets import load_dataset

dataset = load_dataset("klue", "sts")
print(dataset)


exit(0)


tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
# tokenizer.save_pretrained("klue.tok")

tokens = tokenizer.tokenize("안녕하세요. 반갑습니다.")
print(tokens)


vocab = Vocab("klue.tok/vocab.txt")

model = AutoModel.from_pretrained("klue/bert-base")
print(type(model))
# save bert config
save_dir = "klue.bert"
model.config.save_pretrained(save_dir)
# save bert model
model.save_pretrained(save_dir)


shutil.copy(save_dir + "/config.json", "resources/klue/bert_config.json")
shutil.copy(save_dir + "/pytorch_model.bin", "resources/klue/bert_model.pth")

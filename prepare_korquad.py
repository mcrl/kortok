from transformers import AutoTokenizer, AutoModel
import shutil
import os
import requests
from tqdm import tqdm


target_dir = "resources/korquad"
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

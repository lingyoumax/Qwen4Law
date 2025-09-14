from modelscope import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
import torch
import pandas as pd
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import BatchEncoding
import copy
import os
from tqdm.auto import tqdm

from scripts.settings import random_seed, llm_test_ratio, device, llm_modelname, rewardmodel_modelname, llm_max_length, llm_batch_size
from scripts.tools import getReward

lr = 1e-5
K=4
temperature=0.5
n_epoch = 2
savePath = "weight/LLM_GRPO"
os.makedirs(savePath, exist_ok=True)

df = pd.read_csv("data/LLMDataset_RLHF.csv", encoding="utf-8-sig")
def row_to_sample(row):
    query=f"<|im_start|>user\nBased on the content:{row['doc']}\nAnswer the Question:{row['query']}\n/no_think<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return {
        "input": query
    }

dataset = Dataset.from_list([row_to_sample(row) for _, row in df.iterrows()])
dataset_split = dataset.train_test_split(test_size=llm_test_ratio, seed=random_seed)
train_dataset = dataset_split["train"]
test_dataset  = dataset_split["test"]

rewardmodel_adapter_path = "weight/RewardModel_QLoRA"
rewardmodel = AutoModelForCausalLM.from_pretrained(rewardmodel_modelname).to(device)
rewardmodel = PeftModel.from_pretrained(rewardmodel, rewardmodel_adapter_path)
rewardmodel.eval()
rewardmodel_tokenizer = AutoTokenizer.from_pretrained(rewardmodel_modelname, padding_side='left')

token_false_id = rewardmodel_tokenizer.convert_tokens_to_ids("no")
token_true_id = rewardmodel_tokenizer.convert_tokens_to_ids("yes")

llm = AutoModelForCausalLM.from_pretrained(
    llm_modelname,
    device_map=device,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
llm = PeftModel.from_pretrained(
    llm, 
    "weight/LLM_SFT"
).to(device)

for name, param in llm.named_parameters():
    if "lora" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

llm_tokenizer = AutoTokenizer.from_pretrained(llm_modelname, trust_remote_code=True)
llm_tokenizer.padding_side = "left"

'''
reference_llm=copy.deepcopy(llm)
reference_llm.eval()
'''

def tokenize_function(example):
    query = llm_tokenizer(example["input"], padding="max_length", truncation=True, max_length=llm_max_length, return_tensors="pt")
    features ={}
    features["input_ids"]=query["input_ids"][0]
    features["attention_mask"]=query["attention_mask"][0]
    features["query"] = example["input"]

    return features

tokenized_train_dataset = train_dataset.map(tokenize_function, remove_columns=train_dataset.column_names)
tokenized_test_dataset = test_dataset.map(tokenize_function, remove_columns=test_dataset.column_names)

def collate_fn(batch):
    batch_dict = {}

    batch_dict["input_ids"] = torch.stack([torch.tensor(item["input_ids"]) for item in batch]).to(device)
    batch_dict["attention_mask"] = torch.stack([torch.tensor(item["attention_mask"]) for item in batch]).to(device)
    batch_dict["query"]=[item["query"] for item in batch]

    return batch_dict

train_dataloader = DataLoader(
    tokenized_train_dataset,
    batch_size = llm_batch_size,
    shuffle = True,
    collate_fn=collate_fn
)
test_dataloader = DataLoader(
    tokenized_test_dataset,
    batch_size = llm_batch_size,
    shuffle = False,
    collate_fn=collate_fn
)

optimizer = torch.optim.AdamW(llm.parameters(), lr=lr)

for epoch in tqdm(range(n_epoch), desc="Training"):
    trainloss=0
    for batch in tqdm(train_dataloader):
        answers=[[] for _ in range(llm_batch_size)]
        for _ in range(K):
            with torch.no_grad(): 
                input_dict = BatchEncoding({"input_ids":batch["input_ids"],"attention_mask":batch["attention_mask"]})
                generated_ids = llm.generate(**input_dict, max_new_tokens=1024, do_sample=True, temperature=temperature)
                for i in range(llm_batch_size):
                    answer_ids = generated_ids[i][len(batch["input_ids"][i]):].tolist() 
                    answer = llm_tokenizer.decode(answer_ids, skip_special_tokens=True).strip("\n")
                    answers[i].append(answer)
        for i in range(llm_batch_size):            
            query=batch["query"][i]
            r=getReward(rewardmodel,rewardmodel_tokenizer,query,answers[i],token_false_id,token_true_id)
            print(r)
            a=(r-torch.mean(r))/torch.std(r)
            print(a)

llm.save_pretrained(savePath)
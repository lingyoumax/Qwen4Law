from modelscope import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
import torch
import pandas as pd
from peft import PeftModel
from torch.utils.data import DataLoader
import torch
import os

from scripts.settings import random_seed, llm_test_ratio, device, llm_modelname, rewardmodel_modelname
from scripts.tools import getReward

K=4
savePath = "weight/LLM_GRPO"
os.makedirs(savePath, exist_ok=True)

df = pd.read_csv("data/LLMDataset_RLHF.csv", encoding="utf-8-sig")
def row_to_sample(row):
    query=f"Based on the content:{row['doc']}\nAnswer the Question:{row['query']}\n/no_think"
    return {
        "input": query,
        "output": row["answer"]
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
llm_tokenizer = AutoTokenizer.from_pretrained(llm_modelname, trust_remote_code=True)

llm = PeftModel.from_pretrained(
    llm, 
    "weight/LLM_SFT"
).to(device)

for _,row in df.iterrows():
    r=getReward(rewardmodel, rewardmodel_tokenizer, row["query"], [row["answer_good"]], token_false_id, token_true_id)
    print(r)
    break
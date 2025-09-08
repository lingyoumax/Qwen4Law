from modelscope import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
import torch
import pandas as pd
from peft import PeftModel
from torch.utils.data import DataLoader
import torch
import os

from scripts.settings import random_seed, rewardmodel_test_ratio, device, rewardmodel_max_length, llm_modelname
from scripts.tools import getReward

savePath = "weight/LLM_GRPO"
os.makedirs(savePath, exist_ok=True)

df = pd.read_csv("data/LLMDataset_RLHF.csv", encoding="utf-8-sig")

rewardmodel_adapter_path = "weight/RewardModel_QLoRA"
rewardmodel = AutoModelForCausalLM.from_pretrained(rewardmodel_modelname).to(device)
rewardmodel = PeftModel.from_pretrained(rewardmodel, rewardmodel_adapter_path)
rewardmodel.eval()
rewardmodel_tokenizer = AutoTokenizer.from_pretrained(rewardmodel_modelname, padding_side='left')

token_false_id = rewardtokenizer.convert_tokens_to_ids("no")
token_true_id = rewardtokenizer.convert_tokens_to_ids("yes")

base_model = AutoModelForCausalLM.from_pretrained(
    llm_modelname,
    device_map=device,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
llm_tokenizer = AutoTokenizer.from_pretrained(llm_modelname, trust_remote_code=True)

llm = PeftModel.from_pretrained(
    base_model, 
    "weight/LLM_SFT"
).to(device)

for _,row in df.iterrows():
    r=getReward(rewardmodel, rewardtokenizer, row["query"], [row["answer_good"]], token_false_id, token_true_id)
    print(r)
    break
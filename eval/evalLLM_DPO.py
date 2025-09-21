from modelscope import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
import torch
import pandas as pd
from peft import PeftModel
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm
import numpy as np

from scripts.settings import random_seed, llm_test_ratio, device, llm_modelname, rewardmodel_modelname
from scripts.tools import getReward


df = pd.read_csv("data/LLMDataset_RLHF.csv", encoding="utf-8-sig")
def row_to_sample(row):
    query=f"Based on the content:{row['doc']}\nAnswer the Question:{row['query']}\n/no_think"
    return {
        "input": query
    }

dataset = Dataset.from_list([row_to_sample(row) for _, row in df.iterrows()])
dataset_split = dataset.train_test_split(test_size=llm_test_ratio, seed=random_seed)
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
    "weight/LLM_DPO"
).to(device)
llm.eval()

Reward=[]
for sample in tqdm(test_dataset):
    query = sample["input"]
    messages = [
        {"role": "user", "content": query}
    ]
    text = llm_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    model_inputs = llm_tokenizer([text], return_tensors="pt").to(device)
    generated_ids = llm.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    answer = llm_tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    r=getReward(rewardmodel, rewardmodel_tokenizer, query, [answer], token_false_id, token_true_id)
    Reward.append(r)

Reward = torch.cat(Reward, dim=0).tolist()
mean_Reward = np.mean(Reward)
plt.hist(Reward, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
plt.xlim((0,1))
plt.axvline(mean_Reward, color='red', linestyle='--', label=f'Mean: {mean_Reward:.1f}')
plt.title('Distribution of Reward Values')
plt.xlabel('Reward')
plt.ylabel('Count')
plt.legend()

plt.tight_layout()

os.makedirs("figs", exist_ok=True)
plt.savefig('figs/evalLLM_GRPO.svg')
print(mean_Reward)
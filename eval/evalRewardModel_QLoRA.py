from modelscope import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
import torch
import pandas as pd
from peft import PeftModel
from torch.utils.data import DataLoader
import torch

from scripts.settings import random_seed, rewardmodel_test_ratio, device, rewardmodel_modelname, rewardmodel_max_length, rewardmodel_batch_size
from scripts.tools import evaluateTrainedRewardModel

adapter_path = "weight/RewardModel_QLoRA"
df = pd.read_csv("data/LLMDataset_RLHF.csv", encoding="utf-8-sig")

prefix = "<|im_start|>system\nEvaluate the given answer based on the question, and comprehensively assess whether it is a good answer from the perspectives of accuracy, completeness, rigor, usefulness, and natural fluency. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
task = 'Given a legal question, please answer it.'

def format_instruction(query, doc, answer):
    output = f"<Query>: Based on the content:{doc}\nAnswer the Question:{query}\n/no_think\n<Answer>: {answer}"
    return output

def row_to_sample(row):
    return {
        "good_prompt": prefix + format_instruction(row["query"],row['doc'],row["answer_good"]) + suffix,
        "bad_prompt": prefix + format_instruction(row["query"],row['doc'],row["answer_bad"]) + suffix
    }

dataset = Dataset.from_list([row_to_sample(row) for _, row in df.iterrows()])
dataset_split = dataset.train_test_split(test_size=rewardmodel_test_ratio, seed=random_seed)
test_dataset  = dataset_split["test"]

model = AutoModelForCausalLM.from_pretrained(rewardmodel_modelname).to(device)
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(rewardmodel_modelname, padding_side='left')

def tokenize_function(example):
    good_prompt = tokenizer(example["good_prompt"], padding="max_length", truncation=True, max_length=rewardmodel_max_length, return_tensors="pt")
    bad_prompt = tokenizer(example["bad_prompt"], padding="max_length", truncation=True, max_length=rewardmodel_max_length, return_tensors="pt")

    features = {
        "good_prompt_input_ids": good_prompt["input_ids"][0],
        "good_prompt_attention_mask": good_prompt["attention_mask"][0],
        "bad_prompt_input_ids": bad_prompt["input_ids"][0],
        "bad_prompt_attention_mask": bad_prompt["attention_mask"][0]
    }
    return features

tokenized_test_dataset = test_dataset.map(tokenize_function, remove_columns=test_dataset.column_names)

def collate_fn(batch):
    batch_dict = {}

    for key in batch[0]:
        value = torch.stack([torch.tensor(item[key]) for item in batch])
        batch_dict[key] = value.to(device)

    return batch_dict

test_dataloader = DataLoader(
    tokenized_test_dataset,
    batch_size = rewardmodel_batch_size,
    shuffle = False,
    collate_fn=collate_fn
)

token_false_id = tokenizer.convert_tokens_to_ids("no")
token_true_id = tokenizer.convert_tokens_to_ids("yes")
hpc, md, disp = evaluateTrainedRewardModel(model, test_dataloader, token_false_id, token_true_id)
print(hpc)
print(md)
print(disp)
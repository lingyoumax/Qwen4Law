import os
import torch
import pandas as pd
from datasets import Dataset
from modelscope import AutoTokenizer, AutoModelForCausalLM

from settings import llm_modelname, random_seed, llm_test_ratio, llm_max_length

df = pd.read_csv("LLMDataset_SFT.csv", encoding="utf-8-sig")

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

tokenizer = AutoTokenizer.from_pretrained(llm_modelname, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

IGNORE_INDEX = -100

def preprocess_batch(batch):

    messages_prompt = [
        {"role": "user", "content": batch["input"]}
    ]
    messages_full = messages_prompt + [{"role": "assistant", "content": batch["output"]}]

    prompt_ids = tokenizer.apply_chat_template(
        messages_prompt,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors=None
    )
    full_ids = tokenizer.apply_chat_template(
        messages_full,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
        return_tensors=None
    )
    pad_len = 0
    if len(full_ids) > llm_max_length:
        full_trunc = full_ids[-llm_max_length:]
    else:
        pad_len = (llm_max_length - len(full_ids))
        full_trunc = full_ids + [tokenizer.pad_token_id] * pad_len

    p_len_orig = len(prompt_ids)
    trunc_offset = max(0, len(full_ids) - llm_max_length)
    p_len_eff = max(0, p_len_orig - trunc_offset)

    attention_mask = [1]*len(full_trunc)

    labels = full_trunc.copy()
    labels[:p_len_eff] = [IGNORE_INDEX] * p_len_eff

    if pad_len:
        attention_mask[-pad_len:] = [0] * pad_len
        labels[-pad_len:] = [IGNORE_INDEX] * pad_len


    return {
        "input_ids": full_trunc,
        "attention_mask": attention_mask,
        "labels": labels
    }

tokenized_train = train_dataset.map(preprocess_batch, remove_columns=train_dataset.column_names)
tokenized_test  = test_dataset.map(preprocess_batch, remove_columns=test_dataset.column_names)

model = AutoModelForCausalLM.from_pretrained(
    llm_modelname,
    device_map="auto",
    trust_remote_code=True
)


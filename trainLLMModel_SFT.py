import os
from datasets import Dataset
import torch
import pandas as pd
from settings import llm_modelname, device, random_seed, llm_test_ratio

lr=1e-5
n_epoch=10
temperature = 0.05
savePath = "LLMModel_SFT"

if not os.path.exists(savePath):
    os.mkdir(savePath)

df = pd.read_csv("LLMDataset.csv", encoding="utf-8-sig")

def row_to_sample(row):
    sample = {
        "query":  row["query"],
        "answer": row["answer"]
    }
    return sample

data = [row_to_sample(row) for _, row in df.iterrows()]

dataset = Dataset.from_list(data)
dataset_split = dataset.train_test_split(test_size = llm_test_ratio, seed = random_seed)
train_dataset = dataset_split["train"]
test_dataset = dataset_split["test"]
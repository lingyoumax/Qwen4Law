from datasets import Dataset
import pandas as pd
from modelscope import AutoModel, AutoTokenizer
import torch
from torch.utils.data import DataLoader

from settings import num_negative_docs, device, embedding_max_length, random_seed, embedding_modelname, embedding_test_ratio, embedding_batch_size
from tools import evaluateTrainedEmbeddingModel

df = pd.read_csv("RetrieverDataset_selfinstruct_cleaned.csv", encoding="utf-8-sig")

def row_to_sample(row):
    sample = {
        "query": row["query"],
        "positive": row["positive_doc"],
        "negatives": [row[f"negative_doc{i}"] for i in range(num_negative_docs)]
    }
    return sample

data = [row_to_sample(row) for _, row in df.iterrows()]

dataset = Dataset.from_list(data)
dataset_split = dataset.train_test_split(test_size = embedding_test_ratio, seed = random_seed)
test_dataset = dataset_split["test"]

model = AutoModel.from_pretrained(
    embedding_modelname,
    device_map= device
)

model.eval()
tokenizer = AutoTokenizer.from_pretrained(embedding_modelname, padding_side='left')

def tokenize_function(example):
    query = tokenizer(example["query"], padding="max_length", truncation=True, max_length=embedding_max_length, return_tensors="pt")
    positive = tokenizer(example["positive"], padding="max_length", truncation=True, max_length=embedding_max_length, return_tensors="pt")
    negatives = tokenizer(example["negatives"], padding="max_length", truncation=True, max_length=embedding_max_length, return_tensors="pt")

    features = {
        "query_input_ids": query["input_ids"][0],
        "query_attention_mask": query["attention_mask"][0],
        "positive_input_ids": positive["input_ids"][0],
        "positive_attention_mask": positive["attention_mask"][0]
    }

    for i in range(num_negative_docs):
        features[f"negative_input_ids_{i}"] = negatives["input_ids"][i]
        features[f"negative_attention_mask_{i}"] = negatives["attention_mask"][i]

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
    batch_size = embedding_batch_size,
    shuffle = False,
    collate_fn=collate_fn
)

margin = evaluateTrainedEmbeddingModel(model, test_dataloader)
print(margin)
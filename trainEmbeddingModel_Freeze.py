from datasets import Dataset
import pandas as pd
from modelscope import AutoConfig, AutoModel, AutoTokenizer
from transformers import BatchEncoding
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch.nn.functional as F
import os

from settings import num_negative_docs, device, embedding_max_length, random_seed, retriever_modelname, embedding_test_ratio, embedding_batch_size
from tools import last_token_pool, evaluateEmbeddingModel, drawEmbeddingLoss

lr=1e-5
n_epoch=10
temperature = 0.05
savePath = "EmbeddingModel_Freeze"

if not os.path.exists(savePath):
    os.mkdir(savePath)

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
train_dataset = dataset_split["train"]
test_dataset = dataset_split["test"]

model = AutoModel.from_pretrained(
    retriever_modelname,
    device_map= device
)
for param in model.parameters():
    param.requires_grad = False

for layer in model.layers[-2:]:
    for param in layer.parameters():
        param.requires_grad = True

for param in model.norm.parameters():
    param.requires_grad = True

model.train()
tokenizer = AutoTokenizer.from_pretrained(retriever_modelname, padding_side='left')

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

tokenized_train_dataset = train_dataset.map(tokenize_function, remove_columns=train_dataset.column_names)
tokenized_test_dataset = test_dataset.map(tokenize_function, remove_columns=test_dataset.column_names)

def collate_fn(batch):
    batch_dict = {}

    for key in batch[0]:
        value = torch.stack([torch.tensor(item[key]) for item in batch])
        batch_dict[key] = value.to(device)

    return batch_dict

train_dataloader = DataLoader(
    tokenized_train_dataset,
    batch_size = embedding_batch_size,
    shuffle = True,
    collate_fn=collate_fn
)
test_dataloader = DataLoader(
    tokenized_test_dataset,
    batch_size = embedding_batch_size,
    shuffle = True,
    collate_fn=collate_fn
)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

TrainLoss = []
TestLoss = []
TrainLoss.append(evaluateEmbeddingModel(model, train_dataloader, temperature))
TestLoss.append(evaluateEmbeddingModel(model, test_dataloader, temperature))
best_testloss=TestLoss[0]

for epoch in tqdm(range(n_epoch), desc="Training"):
    trainloss=0
    model.train()
    for batch in train_dataloader:
        optimizer.zero_grad()
        query_dict=BatchEncoding({"input_ids":batch["query_input_ids"],"attention_mask":batch["query_attention_mask"]})
        query_output = model(**query_dict)
        query_embedding = last_token_pool(query_output.last_hidden_state, batch["query_attention_mask"])#batch_size * embedding_length
        query_embedding = F.normalize(query_embedding, dim=1)

        positive_dict=BatchEncoding({"input_ids":batch["positive_input_ids"],"attention_mask":batch["positive_attention_mask"]})
        positive_output = model(**positive_dict)
        positive_embedding = last_token_pool(positive_output.last_hidden_state, batch["positive_attention_mask"])#batch_size * embedding_length
        positive_embedding = F.normalize(positive_embedding, dim=1)

        negative_embeddings=[]
        for i in range(num_negative_docs):
            negative_dict_i=BatchEncoding({"input_ids":batch[f"negative_input_ids_{i}"],"attention_mask":batch[f"negative_attention_mask_{i}"]})
            negative_output_i = model(**negative_dict_i)
            negative_embedding_i = last_token_pool(negative_output_i.last_hidden_state, batch[f"negative_attention_mask_{i}"])#batch_size * embedding_length
            negative_embedding_i = F.normalize(negative_embedding_i, dim=1)
            negative_embeddings.append(negative_embedding_i)
        
        B = query_embedding.size(0)

        negatives_stacked = torch.stack(negative_embeddings)

        sim_q_pos = torch.sum(query_embedding * positive_embedding, dim=1) / temperature
        sim_q_neg=torch.sum(query_embedding.unsqueeze(0) * negatives_stacked, dim=2).T / temperature
        sim_q_q=query_embedding @ query_embedding.T / temperature
        sim_pos_dj = positive_embedding @ positive_embedding.T / temperature
        sim_q_dj = query_embedding @ positive_embedding.T /temperature

        m = torch.ones_like(sim_q_q, dtype=torch.bool, device=device)
        diag_indices = torch.arange(sim_q_q.size(0), device=device)
        m[diag_indices, diag_indices] = False
        thresholds = sim_q_pos.unsqueeze(1).expand(-1, B) + 0.1
        too_similar = sim_q_q > thresholds
        m = m & (~too_similar)

        Z = torch.exp(sim_q_pos) + torch.sum(torch.exp(sim_q_neg), dim=1) + torch.sum(m * torch.exp(sim_q_q), dim=1) + torch.sum(m * torch.exp(sim_pos_dj), dim=1)+ torch.sum(m * torch.exp(sim_q_dj), dim=1)
        loss = -torch.mean(torch.log(torch.exp(sim_q_pos) / Z))
        loss.backward()
        optimizer.step()
        trainloss=trainloss+loss.item()

    # === Evaluate ===
    model.eval()
    trainloss=trainloss/len(train_dataloader)
    testloss = evaluateEmbeddingModel(model, test_dataloader, temperature)
    tqdm.write(f"Epoch {epoch}: Training Loss = {trainloss:.4f}, Test loss = {testloss:.4f}")
    TrainLoss.append(trainloss)
    TestLoss.append(testloss)

    if testloss < best_testloss:
        best_testloss = testloss
        torch.save(model.state_dict(), f'{savePath}/{savePath}_Best.pth')

torch.save(model, f'{savePath}/{savePath}_Final.pth')
drawEmbeddingLoss(savePath, TrainLoss, TestLoss)  
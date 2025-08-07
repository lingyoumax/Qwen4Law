from transformers import BatchEncoding
import torch
import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

from settings import device, num_negative_docs

def getFiles(directory='laws', fileend='.txt'):
    txt_files = []
    for filename in os.listdir(directory):
        if filename.endswith(fileend):
            name_without_ext = os.path.splitext(filename)[0]
            txt_files.append(name_without_ext)
    return txt_files

def last_token_pool(last_hidden_states, attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=device), sequence_lengths]
    
def evaluateEmbeddingModel(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            query_dict = BatchEncoding({"input_ids": batch["query_input_ids"], "attention_mask": batch["query_attention_mask"]})
            query_output = model(**query_dict)
            query_embedding = last_token_pool(query_output.last_hidden_state, batch["query_attention_mask"])
            query_embedding = F.normalize(query_embedding, dim=1)

            positive_dict = BatchEncoding({"input_ids": batch["positive_input_ids"], "attention_mask": batch["positive_attention_mask"]})
            positive_output = model(**positive_dict)
            positive_embedding = last_token_pool(positive_output.last_hidden_state, batch["positive_attention_mask"])
            positive_embedding = F.normalize(positive_embedding, dim=1)

            negative_embeddings = []
            for i in range(num_negative_docs):
                neg_dict = BatchEncoding({
                    "input_ids": batch[f"negative_input_ids_{i}"],
                    "attention_mask": batch[f"negative_attention_mask_{i}"]
                })
                neg_output = model(**neg_dict)
                neg_embedding = last_token_pool(neg_output.last_hidden_state, batch[f"negative_attention_mask_{i}"])
                neg_embedding = F.normalize(neg_embedding, dim=1)
                negative_embeddings.append(neg_embedding)

            # 将正负例拼接在一起，计算相似度
            candidates = torch.stack([positive_embedding] + negative_embeddings, dim=1)  # [B, 1+K, D]
            query_embedding = query_embedding.unsqueeze(1)  # [B, 1, D]
            sims = torch.sum(query_embedding * candidates, dim=2)  # [B, 1+K]

            # 预测最相似的是哪一个文档（0 表示 positive）
            preds = sims.argmax(dim=1)
            correct += (preds == 0).sum().item()
            total += sims.size(0)

    model.train()
    return correct / total if total > 0 else 0

def drawEmbeddingLoss(savePath, Loss, Recall):
    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    plt.plot(Loss, color='limegreen')
    plt.xlabel('Epoch')
    plt.ylabel("Loss of Training Set")

    plt.subplot(1, 2, 2)
    plt.plot(Recall, color='darkviolet')
    plt.xlabel('Epoch')
    plt.ylabel("Recall@1 of Test Set")

    plt.savefig(f"{savePath}/drawEmbeddingLoss.svg")
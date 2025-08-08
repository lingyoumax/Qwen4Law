from transformers import BatchEncoding
import torch
import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np

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
    
def evaluateEmbeddingModel(model, dataloader, temperature):
    model.eval()
    l = 0
    total = len(dataloader)
    with torch.no_grad():
        for batch in dataloader:
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
            l = l + loss.item()
            
    model.train()
    return l/total if total > 0 else 0

def drawEmbeddingLoss(saveName, TrainLoss, TestLoss):
    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    plt.plot(np.arange(1, len(TrainLoss) + 1), TrainLoss, color='limegreen')
    plt.xlabel('Epoch')
    plt.xlim(1, None)
    plt.ylabel("Loss of Training Set")

    plt.subplot(1, 2, 2)
    plt.plot(TestLoss, color='darkviolet')
    plt.xlabel('Epoch')
    plt.ylabel("Loss of Test Set")

    plt.savefig(f"Figs/drawEmbeddingLoss_{saveName}.svg")
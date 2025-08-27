from transformers import BatchEncoding
import torch
import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from .settings import device, num_negative_docs

def getFiles(directory='data/laws', fileend='.txt'):
    # 获取文件夹下指定后缀的所有文件的文件名
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
    # 计算模型在测试集上的Loss
    model.eval()
    l = 0
    total = len(dataloader)
    with torch.inference_mode():
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

def evaluateTrainedEmbeddingModel(model, dataloader):
    # 计算模型在测试集上的分离度
    model.eval()
    margin = 0
    total = 0
    with torch.inference_mode():
        for batch in tqdm(dataloader):
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

            sim_q_pos = torch.sum(query_embedding * positive_embedding, dim=1)
            sim_q_negs=torch.sum(query_embedding.unsqueeze(0) * negatives_stacked, dim=2).T
            sim_q_neg=torch.amax(sim_q_negs, dim=1)
            
            margin = margin + torch.sum(sim_q_pos-sim_q_neg).item()
            total = total + B

            
    model.train()
    return margin/total if total > 0 else 0

def computeRerankerScore(model, inputs, token_true_id, token_false_id, ind = 0):
    batch_scores = model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([true_vector, false_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores[:, ind]
    return scores

def evaluateRerankerModel(model, dataloader, token_true_id , token_false_id):
    # 计算模型在测试集上的Loss
    model.eval()
    l = 0
    total = len(dataloader)
    with torch.inference_mode():
        for batch in tqdm(dataloader):
            positive_pair_dict=BatchEncoding({"input_ids":batch["positive_pair_input_ids"],"attention_mask":batch["positive_pair_attention_mask"]})
            scores = computeRerankerScore(model, positive_pair_dict, token_true_id, token_false_id)
        
            for i in range(num_negative_docs):
                negative_pair_dict_i=BatchEncoding({"input_ids":batch[f"negative_pair_input_ids_{i}"],"attention_mask":batch[f"negative_pair_attention_mask_{i}"]})
                negative_pair_score_i = computeRerankerScore(model, negative_pair_dict_i, token_true_id, token_false_id, 1)
                scores = scores + negative_pair_score_i
        
            loss = -torch.mean(scores-torch.log(torch.tensor(1+num_negative_docs)))
            l = l+loss.item()
            
    model.train()
    return l/total if total > 0 else 0

def evaluateTrainedRerankerModel(model, dataloader, token_true_id , token_false_id):
    # 计算模型在测试集上的平均Score
    model.eval()
    score = 0
    total = 0
    with torch.inference_mode():
        for batch in tqdm(dataloader):
            positive_pair_dict=BatchEncoding({"input_ids":batch["positive_pair_input_ids"],"attention_mask":batch["positive_pair_attention_mask"]})
            s = computeRerankerScore(model, positive_pair_dict, token_true_id, token_false_id).exp()
        
            for i in range(num_negative_docs):
                negative_pair_dict_i=BatchEncoding({"input_ids":batch[f"negative_pair_input_ids_{i}"],"attention_mask":batch[f"negative_pair_attention_mask_{i}"]})
                negative_pair_score_i = computeRerankerScore(model, negative_pair_dict_i, token_true_id, token_false_id, 1)
                s = s + negative_pair_score_i.exp()
        
            score = score + torch.sum(s/torch.tensor(1+num_negative_docs)).item()
            B = s.size(0)
            total = total + B

    model.train()
    return score/total if total > 0 else 0

def drawLoss(saveName, TrainLoss, TestLoss):
    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    plt.plot(TrainLoss, color='limegreen')
    plt.xlabel('Epoch')
    plt.ylabel("Loss of Training Set")

    plt.subplot(1, 2, 2)
    plt.plot(TestLoss, color='darkviolet')
    plt.xlabel('Epoch')
    plt.ylabel("Loss of Test Set")

    plt.savefig(f"figs/{saveName}.svg")
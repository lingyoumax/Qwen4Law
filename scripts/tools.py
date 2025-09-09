from transformers import BatchEncoding
import torch
import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np

from .settings import device, num_negative_docs, rewardmodel_max_length

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

def evaluateRewardModel(model, dataloader, token_false_id, token_true_id):
    # 计算模型在测试集上的Loss
    model.eval()
    l = 0
    total = 0
    with torch.inference_mode():
        for batch in tqdm(dataloader):
            good_prompt_dict=BatchEncoding({"input_ids":batch["good_prompt_input_ids"],"attention_mask":batch["good_prompt_attention_mask"]})
            good_batch_scores = model(**good_prompt_dict).logits[:, -1, :]
            good_true_vector = good_batch_scores[:, token_true_id]
            good_false_vector = good_batch_scores[:, token_false_id]
            good_s = good_true_vector - good_false_vector

            median_prompt_dict=BatchEncoding({"input_ids":batch["median_prompt_input_ids"],"attention_mask":batch["median_prompt_attention_mask"]})
            median_batch_scores = model(**median_prompt_dict).logits[:, -1, :]
            median_true_vector = median_batch_scores[:, token_true_id]
            median_false_vector = median_batch_scores[:, token_false_id]
            median_s = median_true_vector - median_false_vector

            bad_prompt_dict=BatchEncoding({"input_ids":batch["bad_prompt_input_ids"],"attention_mask":batch["bad_prompt_attention_mask"]})
            bad_batch_scores = model(**bad_prompt_dict).logits[:, -1, :]
            bad_true_vector = bad_batch_scores[:, token_true_id]
            bad_false_vector = bad_batch_scores[:, token_false_id]
            bad_s = bad_true_vector - bad_false_vector

            B = bad_s.size(0)
            good_loss = -torch.sum(torch.log(torch.sigmoid(good_s)))
            median_loss = -torch.sum(torch.log(1-torch.abs(0.5-torch.sigmoid(median_s))))
            bad_loss = -torch.sum(torch.log(1-torch.sigmoid(bad_s)))
            good_median_loss = -torch.sum(torch.log(torch.clamp(torch.sigmoid(good_s)-torch.sigmoid(median_s),min=1e-8,max=0.5)))
            median_bad_loss = -torch.sum(torch.log(torch.clamp(torch.sigmoid(median_s)-torch.sigmoid(bad_s),min=1e-8,max=0.5)))
            loss=good_loss+median_loss+bad_loss+good_median_loss+median_bad_loss
            l = l + loss.item()
            total = total + B
            
    model.train()
    return l/total if total > 0 else 0

def evaluateTrainedRewardModel(model, dataloader, token_false_id, token_true_id):
    # 计算模型在测试集上的HPC（人类偏好一致性）、MD（均值差）、Disp（合并标准差）
    model.eval()
    total = 0
    hpc_count = 0
    good_scores=[]
    median_scores=[]
    bad_scores=[]
    with torch.inference_mode():
        for batch in tqdm(dataloader):
            good_prompt_dict=BatchEncoding({"input_ids":batch["good_prompt_input_ids"],"attention_mask":batch["good_prompt_attention_mask"]})
            good_batch_scores = model(**good_prompt_dict).logits[:, -1, :]
            good_true_vector = good_batch_scores[:, token_true_id]
            good_false_vector = good_batch_scores[:, token_false_id]
            good_batch_scores = torch.stack([good_false_vector, good_true_vector], dim=1)
            good_score=torch.nn.functional.softmax(good_batch_scores, dim=1)[:, 1]

            median_prompt_dict=BatchEncoding({"input_ids":batch["median_prompt_input_ids"],"attention_mask":batch["median_prompt_attention_mask"]})
            median_batch_scores = model(**median_prompt_dict).logits[:, -1, :]
            median_true_vector = median_batch_scores[:, token_true_id]
            median_false_vector = median_batch_scores[:, token_false_id]
            median_batch_scores = torch.stack([median_false_vector, median_true_vector], dim=1)
            median_score=torch.nn.functional.softmax(median_batch_scores, dim=1)[:, 1]

            bad_prompt_dict=BatchEncoding({"input_ids":batch["bad_prompt_input_ids"],"attention_mask":batch["bad_prompt_attention_mask"]})
            bad_batch_scores = model(**bad_prompt_dict).logits[:, -1, :]
            bad_true_vector = bad_batch_scores[:, token_true_id]
            bad_false_vector = bad_batch_scores[:, token_false_id]
            bad_batch_scores = torch.stack([bad_false_vector, bad_true_vector], dim=1)
            bad_score=torch.nn.functional.softmax(bad_batch_scores, dim=1)[:, 1]

            B = good_score.size(0)
            hpc_count += torch.logical_and(good_score > median_score, median_score > bad_score).int().sum().item()
            total = total + B
            good_scores.extend(good_score.tolist())
            median_scores.extend(median_score.tolist())
            bad_scores.extend(bad_score.tolist())
            
    model.train()
    hpc = hpc_count / total if total > 0 else 0.0
    good_scores_np = np.array(good_scores)
    median_scores_np = np.array(median_scores)
    bad_scores_np = np.array(bad_scores)
    
    mean_good = np.mean(good_scores_np)
    mean_median = np.mean(median_scores_np)
    mean_bad = np.mean(bad_scores_np)
    md_good_median = mean_good - mean_median
    md_median_bad = mean_median - mean_bad

    std_good = np.std(good_scores_np)
    std_median = np.std(median_scores_np)
    std_bad = np.std(bad_scores_np)
    disp = np.sqrt((std_good**2 + std_bad**2) / 2)
    disp = np.sqrt((std_good**2 + std_median**2 + std_bad**2) / 3)

    return hpc, md_good_median, md_median_bad, disp

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

@torch.no_grad()
def getReward(model, tokenizer, query,answers, token_false_id, token_true_id):
    task = 'Given a legal question, please answer it.'
    system = "Evaluate the given answer based on the question, and comprehensively assess whether it is a good answer from the perspectives of accuracy, completeness, rigor, usefulness, and natural fluency.Note that the answer can only be \"yes\" or \"no\"."

    def build_prompt(ans: str) -> str:
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n<Instruct>: {task}\n<Query>: {query}\n<Answer>: {ans}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )

    prompts = [build_prompt(a) for a in answers]
    enc = tokenizer(
        prompts, padding=True, truncation=True, max_length=rewardmodel_max_length, return_tensors="pt"
    ).to(model.device)
    scores = model(**enc).logits[:, -1, :]
    true_vector = scores[:, token_true_id]
    false_vector = scores[:, token_false_id]
    r = torch.sigmoid(true_vector-false_vector)
    return r
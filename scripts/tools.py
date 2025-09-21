from transformers import BatchEncoding
import torch
from torch.amp import autocast
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

            bad_prompt_dict=BatchEncoding({"input_ids":batch["bad_prompt_input_ids"],"attention_mask":batch["bad_prompt_attention_mask"]})
            bad_batch_scores = model(**bad_prompt_dict).logits[:, -1, :]
            bad_true_vector = bad_batch_scores[:, token_true_id]
            bad_false_vector = bad_batch_scores[:, token_false_id]
            bad_s = bad_true_vector - bad_false_vector

            B = bad_s.size(0)
            good_loss = -torch.sum(torch.log(torch.sigmoid(good_s)))
            bad_loss = -torch.sum(torch.log(1-torch.sigmoid(bad_s)))
            loss=good_loss+bad_loss
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
    bad_scores=[]
    with torch.inference_mode():
        for batch in tqdm(dataloader):
            good_prompt_dict=BatchEncoding({"input_ids":batch["good_prompt_input_ids"],"attention_mask":batch["good_prompt_attention_mask"]})
            good_batch_scores = model(**good_prompt_dict).logits[:, -1, :]
            good_true_vector = good_batch_scores[:, token_true_id]
            good_false_vector = good_batch_scores[:, token_false_id]
            good_batch_scores = torch.stack([good_false_vector, good_true_vector], dim=1)
            good_score=torch.nn.functional.softmax(good_batch_scores, dim=1)[:, 1]

            bad_prompt_dict=BatchEncoding({"input_ids":batch["bad_prompt_input_ids"],"attention_mask":batch["bad_prompt_attention_mask"]})
            bad_batch_scores = model(**bad_prompt_dict).logits[:, -1, :]
            bad_true_vector = bad_batch_scores[:, token_true_id]
            bad_false_vector = bad_batch_scores[:, token_false_id]
            bad_batch_scores = torch.stack([bad_false_vector, bad_true_vector], dim=1)
            bad_score=torch.nn.functional.softmax(bad_batch_scores, dim=1)[:, 1]

            B = good_score.size(0)
            hpc_count += (good_score > bad_score).int().sum().item()
            total = total + B
            good_scores.extend(good_score.tolist())
            bad_scores.extend(bad_score.tolist())
            
    model.train()
    hpc = hpc_count / total if total > 0 else 0.0
    good_scores_np = np.array(good_scores)
    bad_scores_np = np.array(bad_scores)
    
    mean_good = np.mean(good_scores_np)
    mean_bad = np.mean(bad_scores_np)
    md = mean_good - mean_bad

    std_good = np.std(good_scores_np)
    std_bad = np.std(bad_scores_np)
    disp = np.sqrt((std_good**2 + std_bad**2) / 2)

    return hpc, md, disp

@torch.inference_mode()
def evaluateLLM_DPO(model, dataloader, beta):

    def _get_response_logp(model, input_ids, attention_mask, prompt_lens):
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    
        log_probs = F.log_softmax(logits, dim=-1)
    
        labels = input_ids[:, 1:]
        log_probs = log_probs[:, :-1, :]
    
        B, L, V = log_probs.shape
        token_log_probs = log_probs.gather(
            dim=-1,
            index=labels.unsqueeze(-1)
        ).squeeze(-1)
    
        B, L_minus_1 = token_log_probs.shape
        arange = torch.arange(L_minus_1, device=input_ids.device) 
        mask = (arange.unsqueeze(0) >= prompt_lens.unsqueeze(1)).float() 
    
        mask = mask * attention_mask[:, 1:].float() 
    
        sum_logp = (token_log_probs * mask).sum(dim=1)
        count_tokens = mask.sum(dim=1)
    
        count_tokens = torch.clamp(count_tokens, min=1.0)
        avg_logp = sum_logp / count_tokens
    
        return avg_logp

    model.eval()
    l = 0
    total = 0
    for batch in tqdm(dataloader):
        logp_chosen = _get_response_logp(
            model,
            input_ids=batch["chosen_ids"],
            attention_mask=batch["chosen_attention_mask"],
            prompt_lens=batch["prompt_lens"]
        )

        logp_rejected = _get_response_logp(
            model,
            input_ids=batch["rejected_ids"],
            attention_mask=batch["rejected_attention_mask"],
            prompt_lens=batch["prompt_lens"]
        )
        loss = -F.logsigmoid(beta * (logp_chosen - logp_rejected)).sum()
        B=batch["chosen_ids"].shape[0]
        l+=loss.item()
        total+=B
            
    model.train()
    return l/total if total > 0 else 0

@torch.inference_mode()
def evaluateTrainedLLM(llm,llm_tokenizer,rewardmodel,rewardmodel_tokenizer,dataloader,token_false_id,token_true_id,max_new_tokens):
    """
    Faster evaluation:
    1) batched generation & decode
    2) batched reward scoring
    3) correct last-token gather
    """

    system = (
        'Evaluate the given answer based on the question, and comprehensively assess '
        'whether it is a good answer from the perspectives of accuracy, completeness, '
        'rigor, usefulness, and natural fluency. Note that the answer can only be "yes" or "no".'
    )

    def build_prompt(query, answer):
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n<Query>: {query}\n<Answer>: {answer}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )

    llm.eval()
    rewardmodel.eval()

    total_reward = 0.0
    total_count = 0

    # 尽量在 GPU 上生成 + 自动混精
    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(llm.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(llm.device, non_blocking=True)

        # ===== 1) 一次性生成 =====
        with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            gen_ids = llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=llm_tokenizer.pad_token_id,
                eos_token_id=llm_tokenizer.eos_token_id,
            )

        # ===== 2) 批量 decode =====
        # 去掉 prompt 部分
        cut_ids = [
            g[len(inp):] for g, inp in zip(gen_ids, input_ids)
        ]
        answers = llm_tokenizer.batch_decode(cut_ids, skip_special_tokens=True)
        # 去除多余空格/换行
        answers = [a.strip() for a in answers]

        # ===== 3) 构造 reward prompts =====
        prompts = [build_prompt(q, a) for q, a in zip(batch["query"], answers)]

        # ===== 4) 奖励模型一次性推理 =====
        enc = rewardmodel_tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=rewardmodel_max_length,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        input_ids_r = enc["input_ids"].to(rewardmodel.device, non_blocking=True)
        attention_mask_r = enc["attention_mask"].to(rewardmodel.device, non_blocking=True)

        with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            logits = rewardmodel(
                input_ids=input_ids_r,
                attention_mask=attention_mask_r
            ).logits                                   # [N, L, V]

        # 取每条样本最后一个有效 token 的 logits
        last_idx = attention_mask_r.sum(dim=1) - 1
        batch_idx = torch.arange(logits.size(0), device=logits.device)
        last_logits = logits[batch_idx, last_idx, :]   # [N, V]

        r = torch.sigmoid(
            last_logits[:, token_true_id] - last_logits[:, token_false_id]
        )

        total_reward += r.sum().item()
        total_count  += r.size(0)

    llm.train()
    return total_reward / max(total_count, 1)

@torch.no_grad()
def getReward(model, tokenizer, query, answers, token_false_id, token_true_id):
    system = "Evaluate the given answer based on the question, and comprehensively assess whether it is a good answer from the perspectives of accuracy, completeness, rigor, usefulness, and natural fluency. Note that the answer can only be \"yes\" or \"no\"."

    def build_prompt(answer: str) -> str:
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n<Query>: {query}\n<Answer>: {answer}<|im_end|>\n"
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

@torch.no_grad()
def getReward_batched(model, tokenizer,
                      queries: list[str],
                      answers: list[list[str]],
                      token_false_id: int,
                      token_true_id: int) -> torch.Tensor:
    """
    queries : list of length B
    answers : list of length B, each a list of K answers
    returns : tensor [B, K] of rewards in [0,1]
    """
    system = (
        "Evaluate the given answer based on the question, and comprehensively "
        "assess whether it is a good answer from the perspectives of accuracy, "
        "completeness, rigor, usefulness, and natural fluency. "
        'Note that the answer can only be "yes" or "no".'
    )

    # Build all prompts in a single flat list
    flat_prompts = []
    for q, ans_list in zip(queries, answers):
        for a in ans_list:
            prompt = (
                f"<|im_start|>system\n{system}<|im_end|>\n"
                f"<|im_start|>user\n<Query>: {q}\n<Answer>: {a}<|im_end|>\n"
                f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
            )
            flat_prompts.append(prompt)

    # Tokenize all at once
    enc = tokenizer(
        flat_prompts,
        padding=True,
        truncation=True,
        max_length=rewardmodel_max_length,
        return_tensors="pt"
    ).to(model.device)

    # Single forward pass
    logits = model(**enc).logits[:, -1, :]
    r = torch.sigmoid(logits[:, token_true_id] - logits[:, token_false_id])

    # Reshape back to [B, K]
    B, K = len(queries), len(answers[0])
    return r.view(B, K)

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

def drawReward(saveName, TrainR, TestR):
    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    plt.plot(TrainR, color='limegreen')
    plt.xlabel('Epoch')
    plt.ylabel("Reward of Training Set")

    plt.subplot(1, 2, 2)
    plt.plot(TestR, color='darkviolet')
    plt.xlabel('Epoch')
    plt.ylabel("Reward of Test Set")

    plt.savefig(f"figs/{saveName}.svg")
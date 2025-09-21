import os
import torch
import pandas as pd
from tqdm.auto import tqdm
from datasets import Dataset
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import BatchEncoding
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F
from modelscope import AutoModelForCausalLM, AutoTokenizer 
import warnings

from scripts.settings import random_seed, llm_test_ratio, device, llm_modelname, llm_max_length, rlhf_llm_batch_size
from scripts.tools import evaluateLLM_DPO, drawLoss 

warnings.filterwarnings("ignore")

lr = 1e-5
n_epoch = 5
max_new_tokens = 512
beta = 1
grad_clip = 1.0
savePath = "weight/LLM_DPO"
os.makedirs(savePath, exist_ok=True)

df = pd.read_csv("data/LLMDataset_RLHF.csv", encoding="utf-8-sig")

def row_to_sample(row):
    """构建prompt+chosen/rejected的完整文本（匹配模型对话格式）"""
    prompt = (
        f"<|im_start|>user\nBased on the content:{row['doc']}\n"
        f"Answer the Question:{row['query']}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    chosen_text = prompt + row["answer_good"].strip() + f"\n<|im_end|>"
    rejected_text = prompt + row["answer_bad"].strip() + f"\n<|im_end|>"
    query=f"Based on the content:{row['doc']}\nAnswer the Question:{row['query']}"
    return {
        "prompt": prompt,
        "chosen_text": chosen_text,
        "rejected_text": rejected_text,
        "query":query
    }

dataset = Dataset.from_list([row_to_sample(row) for _, row in df.iterrows()])
dataset_split = dataset.train_test_split(test_size=llm_test_ratio, seed=random_seed)
train_dataset = dataset_split["train"]
test_dataset = dataset_split["test"]

llm = AutoModelForCausalLM.from_pretrained(
    llm_modelname,
    torch_dtype=torch.bfloat16
)
llm = PeftModel.from_pretrained(llm, "weight/LLM_SFT").to(device)

for n, p in llm.named_parameters():
    p.requires_grad = ("lora" in n)

llm_tokenizer = AutoTokenizer.from_pretrained(llm_modelname, trust_remote_code=True)
llm_tokenizer.padding_side = "left"
if llm_tokenizer.pad_token is None:
    llm_tokenizer.pad_token = llm_tokenizer.eos_token

def tokenize_function(example):
    
    prompt = example["prompt"]
    
    prompt_tokenized = llm_tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=llm_max_length,
        return_tensors="pt"
    )

    chosen_tokenized = llm_tokenizer(
        example["chosen_text"],
        padding="max_length",
        truncation=True,
        max_length=llm_max_length,
        return_tensors="pt"
    )
    rejected_tokenized = llm_tokenizer(
        example["rejected_text"],
        padding="max_length",
        truncation=True,
        max_length=llm_max_length,
        return_tensors="pt"
    )
    
    prompt_raw_ids = llm_tokenizer(prompt, truncation=False)["input_ids"]
    prompt_len = len(prompt_raw_ids)
    
    return {
        "input":example["prompt"],
        "input_ids": prompt_tokenized["input_ids"][0],
        "attention_mask": prompt_tokenized["attention_mask"][0],
        "query": example["query"],
        "chosen_ids": chosen_tokenized["input_ids"][0],
        "chosen_attention_mask": chosen_tokenized["attention_mask"][0],
        "rejected_ids": rejected_tokenized["input_ids"][0],
        "rejected_attention_mask": rejected_tokenized["attention_mask"][0],
        "prompt_len": prompt_len
    }

tokenized_train_dataset = train_dataset.map(
    tokenize_function,
    remove_columns=train_dataset.column_names
)
tokenized_test_dataset = test_dataset.map(
    tokenize_function,
    remove_columns=test_dataset.column_names
)

def collate_fn(batch):
    out = {}
    out["input_ids"] = torch.stack([torch.tensor(x["input_ids"]) for x in batch]).to(device, non_blocking=True)
    out["attention_mask"] = torch.stack([torch.tensor(x["attention_mask"]) for x in batch]).to(device, non_blocking=True)
    out["input"] = [x["input"] for x in batch]
    out["query"] = [x["query"] for x in batch]
    out["chosen_ids"] = torch.stack([torch.tensor(x["chosen_ids"]) for x in batch]).to(device, non_blocking=True)
    out["chosen_attention_mask"] = torch.stack([torch.tensor(x["chosen_attention_mask"]) for x in batch]).to(device, non_blocking=True)
    out["rejected_ids"] = torch.stack([torch.tensor(x["rejected_ids"]) for x in batch]).to(device, non_blocking=True)
    out["rejected_attention_mask"] = torch.stack([torch.tensor(x["rejected_attention_mask"]) for x in batch]).to(device, non_blocking=True)
    out["prompt_lens"] = torch.stack([torch.tensor(x["prompt_len"]) for x in batch]).to(device, non_blocking=True)
    return out

train_dataloader = DataLoader(tokenized_train_dataset, batch_size=rlhf_llm_batch_size, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(tokenized_test_dataset, batch_size=rlhf_llm_batch_size, shuffle=False, collate_fn=collate_fn)

def dpo_loss(model, batch, beta):
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
    
    loss = -F.logsigmoid(beta * (logp_chosen - logp_rejected)).mean()
    return loss

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

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, llm.parameters()), lr=lr)

llm.train()
TrainLoss = []
TestLoss = []
TrainLoss.append(evaluateLLM_DPO(llm, train_dataloader, beta))
TestLoss.append(evaluateLLM_DPO(llm, test_dataloader, beta))
best_testloss = TestLoss[0]

accumulation_steps = 4
global_step = 0

for epoch in tqdm(range(n_epoch)):
    epoch_loss = 0.0
    accum_step = 0

    for batch in tqdm(train_dataloader):
        loss = dpo_loss(llm, batch, beta) / accumulation_steps  # 除以累积步数，保证梯度尺度
        
        loss.backward()
        epoch_loss += loss.item() * accumulation_steps 
        accum_step += 1

        if accum_step % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(llm.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        del batch, loss
        torch.cuda.empty_cache()

    if accum_step % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(llm.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

    print(f"[Epoch {epoch+1}] avg loss = {epoch_loss/len(train_dataloader):.4f}")

    trainloss = evaluateLLM_DPO(llm, train_dataloader, beta)
    testloss = evaluateLLM_DPO(llm, test_dataloader, beta)
    
    if testloss < best_testloss:
        best_testloss = testloss
        llm.save_pretrained(savePath)
    
    TrainLoss.append(trainloss)
    TestLoss.append(testloss)

drawLoss("LLM_DPO", TrainLoss, TestLoss)
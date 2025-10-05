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
beta = 0.1
grad_clip = 1.0
savePath = "weight/LLM_DPO"
os.makedirs(savePath, exist_ok=True)

df = pd.read_csv("data/LLMDataset_RLHF.csv", encoding="utf-8-sig")

def row_to_sample(row):
    prompt = (
        f"<|im_start|>user\nBased on the content:{row['doc']}\n"
        f"Answer the Question:{row['query']}\n/no_think<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    chosen_text = prompt + row["answer_good"] + f"\n<|im_end|>"
    rejected_text = prompt + row["answer_bad"] + f"\n<|im_end|>"
    return {
        "prompt": prompt,
        "chosen_text": chosen_text,
        "rejected_text": rejected_text
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

llm_tokenizer = AutoTokenizer.from_pretrained(llm_modelname)
llm_tokenizer.padding_side = "left"
if llm_tokenizer.pad_token is None:
    llm_tokenizer.pad_token = llm_tokenizer.eos_token

def tokenize_function(example):
    prompt = example["prompt"]
    
    prompt_ids_trunc = llm_tokenizer(
        prompt, truncation=True, max_length=llm_max_length, add_special_tokens=True
    )["input_ids"]
    prompt_len_trunc = len(prompt_ids_trunc)

    # 2) 编码 chosen/rejected（保持你原来的设定）
    chosen_tok = llm_tokenizer(
        example["chosen_text"], padding="max_length", truncation=True,
        max_length=llm_max_length, return_tensors="pt"
    )
    rejected_tok = llm_tokenizer(
        example["rejected_text"], padding="max_length", truncation=True,
        max_length=llm_max_length, return_tensors="pt"
    )

    # 3) 各自的左侧 pad 数（left padding -> attention_mask 前面是 0）
    chosen_am = chosen_tok["attention_mask"][0]
    rejected_am = rejected_tok["attention_mask"][0]
    pad_left_chosen = int((chosen_am == 0).sum().item())
    pad_left_rejected = int((rejected_am == 0).sum().item())

    # 4) 各自答案起点（在 input_ids 里的下标）
    L_chosen = int(chosen_tok["input_ids"].shape[1])
    L_rejected = int(rejected_tok["input_ids"].shape[1])
    ans_start_chosen = min(pad_left_chosen + prompt_len_trunc, L_chosen)  # 允许等于 L -> 空掩码
    ans_start_rejected = min(pad_left_rejected + prompt_len_trunc, L_rejected)

    return {

        "chosen_ids": chosen_tok["input_ids"][0],
        "chosen_attention_mask": chosen_am,
        "rejected_ids": rejected_tok["input_ids"][0],
        "rejected_attention_mask": rejected_am,

        "ans_start_chosen": ans_start_chosen,
        "ans_start_rejected": ans_start_rejected,
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
    out["chosen_ids"] = torch.stack([torch.tensor(x["chosen_ids"]) for x in batch]).to(device, non_blocking=True)
    out["chosen_attention_mask"] = torch.stack([torch.tensor(x["chosen_attention_mask"]) for x in batch]).to(device, non_blocking=True)
    out["rejected_ids"] = torch.stack([torch.tensor(x["rejected_ids"]) for x in batch]).to(device, non_blocking=True)
    out["rejected_attention_mask"] = torch.stack([torch.tensor(x["rejected_attention_mask"]) for x in batch]).to(device, non_blocking=True)
    out["ans_start_chosen"] = torch.tensor([x["ans_start_chosen"] for x in batch]).to(device, non_blocking=True)
    out["ans_start_rejected"] = torch.tensor([x["ans_start_rejected"] for x in batch]).to(device, non_blocking=True)
    return out

train_dataloader = DataLoader(tokenized_train_dataset, batch_size=rlhf_llm_batch_size, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(tokenized_test_dataset, batch_size=rlhf_llm_batch_size, shuffle=False, collate_fn=collate_fn)

def dpo_loss(model, batch, beta):
    logp_chosen = _get_response_logp(
        model, batch["chosen_ids"], batch["chosen_attention_mask"],
        batch["ans_start_chosen"]
    )
    logp_rejected = _get_response_logp(
        model, batch["rejected_ids"], batch["rejected_attention_mask"],
        batch["ans_start_rejected"]
    )
    return -F.logsigmoid(beta * (logp_chosen - logp_rejected)).mean()

def _get_response_logp(model, input_ids, attention_mask, answer_starts):
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    log_probs = F.log_softmax(logits, dim=-1)

    labels = input_ids[:, 1:]
    log_probs = log_probs[:, :-1, :]  # 对齐

    token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    B, Lm1 = token_log_probs.shape
    arange = torch.arange(Lm1, device=input_ids.device).unsqueeze(0)  # [1, L-1]

    thresholds = (answer_starts - 1).unsqueeze(1).clamp_min(0)       # [B,1]
    mask_prompt = (arange >= thresholds).float()                     # [B, L-1]
    mask = mask_prompt * attention_mask[:, 1:].float()

    sum_logp = (token_log_probs * mask).sum(dim=1)
    return sum_logp

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, llm.parameters()), lr=lr)

llm.train()
TrainLoss = []
TestLoss = []
TrainLoss.append(evaluateLLM_DPO(llm, train_dataloader, beta,_get_response_logp))
TestLoss.append(evaluateLLM_DPO(llm, test_dataloader, beta,_get_response_logp))
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

    if accum_step % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(llm.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

    print(f"[Epoch {epoch+1}] avg loss = {epoch_loss/len(train_dataloader):.4f}")

    trainloss = evaluateLLM_DPO(llm, train_dataloader, beta, _get_response_logp)
    testloss = evaluateLLM_DPO(llm, test_dataloader, beta, _get_response_logp)
    
    if testloss < best_testloss:
        best_testloss = testloss
        llm.save_pretrained(savePath)
    
    TrainLoss.append(trainloss)
    TestLoss.append(testloss)

drawLoss("LLM_DPO", TrainLoss, TestLoss)
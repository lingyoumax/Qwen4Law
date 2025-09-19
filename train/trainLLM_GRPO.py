import os
import torch
import pandas as pd
from tqdm.auto import tqdm
from datasets import Dataset
from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
from torch.utils.data import DataLoader
from transformers import BatchEncoding
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F
from modelscope import AutoModelForCausalLM, AutoTokenizer 

from scripts.settings import random_seed, llm_test_ratio, device, llm_modelname, rewardmodel_modelname, llm_max_length, rlhf_llm_batch_size
from scripts.tools import getReward, evaluateTrainedLLM, drawReward

lr = 1e-5
K = 4
temperature = 0.7
n_epoch = 5
max_new_tokens = 512
kl_coef = 0.02
grad_clip = 1.0
savePath = "weight/LLM_GRPO"
os.makedirs(savePath, exist_ok=True)

df = pd.read_csv("data/LLMDataset_RLHF.csv", encoding="utf-8-sig")

def row_to_sample(row):
    text = (
        f"<|im_start|>user\nBased on the content:{row['doc']}\n"
        f"Answer the Question:{row['query']}\n/no_think<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    query=f"Based on the content:{row['doc']}\nAnswer the Question:{row['query']}"
    return {"input": text,"query":query}

dataset = Dataset.from_list([row_to_sample(row) for _, row in df.iterrows()])
dataset_split = dataset.train_test_split(test_size=llm_test_ratio, seed=random_seed)
train_dataset = dataset_split["train"]
test_dataset = dataset_split["test"]

rewardmodel_adapter_path = "weight/RewardModel_QLoRA"
rewardmodel = AutoModelForCausalLM.from_pretrained(rewardmodel_modelname, trust_remote_code=True)
rewardmodel = PeftModel.from_pretrained(rewardmodel, rewardmodel_adapter_path)
rewardmodel.eval().to(device)

rewardmodel_tokenizer = AutoTokenizer.from_pretrained(rewardmodel_modelname, padding_side="left", trust_remote_code=True)
if rewardmodel_tokenizer.pad_token is None:
    rewardmodel_tokenizer.pad_token = rewardmodel_tokenizer.eos_token

token_false_id = rewardmodel_tokenizer.convert_tokens_to_ids("no")
token_true_id  = rewardmodel_tokenizer.convert_tokens_to_ids("yes")


llm = AutoModelForCausalLM.from_pretrained(
    llm_modelname,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
llm = PeftModel.from_pretrained(llm, "weight/LLM_SFT").to(device)

for n, p in llm.named_parameters():
    p.requires_grad = ("lora" in n)

llm_tokenizer = AutoTokenizer.from_pretrained(llm_modelname, trust_remote_code=True)
llm_tokenizer.padding_side = "left"
if llm_tokenizer.pad_token is None:
    llm_tokenizer.pad_token = llm_tokenizer.eos_token

reference_llm = AutoModelForCausalLM.from_pretrained(
    llm_modelname,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
reference_llm = PeftModel.from_pretrained(reference_llm, "weight/LLM_SFT").to(device)
reference_llm.eval()
for param in reference_llm.parameters():
    param.requires_grad = False

def tokenize_function(example):
    q = llm_tokenizer(
        example["input"],
        padding="max_length",
        truncation=True,
        max_length=llm_max_length,
        return_tensors="pt",
    )
    return {
        "input_ids": q["input_ids"][0],
        "attention_mask": q["attention_mask"][0],
        "input": example["input"],
        "query": example["query"],
    }

tokenized_train_dataset = train_dataset.map(tokenize_function, remove_columns=train_dataset.column_names)
tokenized_test_dataset  = test_dataset.map(tokenize_function,  remove_columns=test_dataset.column_names)

def collate_fn(batch):
    out = {}
    out["input_ids"] = torch.stack([torch.tensor(x["input_ids"]) for x in batch]).to(device)
    out["attention_mask"] = torch.stack([torch.tensor(x["attention_mask"]) for x in batch]).to(device)
    out["input"] = [x["input"] for x in batch]
    out["query"] = [x["query"] for x in batch]
    return out

train_dataloader = DataLoader(tokenized_train_dataset, batch_size=rlhf_llm_batch_size, shuffle=True,  collate_fn=collate_fn)
test_dataloader  = DataLoader(tokenized_test_dataset,  batch_size=rlhf_llm_batch_size, shuffle=False, collate_fn=collate_fn)


optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, llm.parameters()), lr=lr)

@torch.no_grad()
def sample_answers(model, input_ids, attention_mask, K, temperature, max_new_tokens):
    model.eval()
    B = input_ids.size(0)
    answers_text = [[] for _ in range(B)]
    answers_ids  = [[] for _ in range(B)]
    for _ in range(K):
        inps = BatchEncoding({"input_ids": input_ids, "attention_mask": attention_mask})
        gen_ids = model.generate(
            **inps,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            pad_token_id=llm_tokenizer.pad_token_id,
            eos_token_id=llm_tokenizer.eos_token_id,
        )
        for i in range(B):
            ans_ids = gen_ids[i][input_ids[i].shape[0]:].tolist()
            text = llm_tokenizer.decode(ans_ids, skip_special_tokens=True).strip()
            answers_text[i].append(text)
            answers_ids[i].append(ans_ids)
        del gen_ids
    model.train()
    return answers_text, answers_ids

def build_concat_batch(prompts_ids, prompts_mask, answers_ids_list, tokenizer, max_len=None):
    B = prompts_ids.size(0)
    seqs, attns, prompt_lens, ans_lens = [], [], [], []
    pad_id = tokenizer.pad_token_id

    for i in range(B):
        m = prompts_mask[i].sum().item()
        prompt = prompts_ids[i][-m:].tolist()

        for ans in answers_ids_list[i]:
            ans = list(ans)
            if len(ans) == 0 or ans[-1] != tokenizer.eos_token_id:
                ans = ans + [tokenizer.eos_token_id]

            seq = prompt + ans
            if max_len is not None and len(seq) > max_len:
                seq = seq[-max_len:]
                pr_len = min(m, len(seq) - len(ans))
            else:
                pr_len = len(prompt)

            att = [1] * len(seq)
            seqs.append(torch.tensor(seq, dtype=torch.long))
            attns.append(torch.tensor(att, dtype=torch.long))
            prompt_lens.append(pr_len)
            ans_lens.append(len(seq) - pr_len)
    concat_input_ids = pad_sequence(seqs, batch_first=True, padding_value=pad_id)
    concat_attention = pad_sequence(attns, batch_first=True, padding_value=0)
    return concat_input_ids, concat_attention, prompt_lens, ans_lens

def sequence_logprobs(model, input_ids, attention_mask, prompt_lens):
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  
    labels = input_ids[:, 1:]                                

    N, Lm1 = labels.shape
    out_logp = []
    out_len = []
    for i in range(N):
        pr_len = prompt_lens[i]
        mask = torch.zeros(Lm1, dtype=torch.bool, device=labels.device)
        
        mask[pr_len:] = True

        token_logp = log_probs[i].gather(-1, labels[i].unsqueeze(-1)).squeeze(-1)
        token_logp = token_logp[mask]
        length = token_logp.numel()
        out_len.append(length)
        if length == 0:
            out_logp.append(torch.tensor(0.0, device=labels.device))
        else:
            out_logp.append(token_logp.mean())#对比求和还是mean更好
    return torch.stack(out_logp, dim=0), torch.tensor(out_len, device=labels.device)

llm.train()

TrainR=[]
TestR=[]

TrainR.append(evaluateTrainedLLM(llm,llm_tokenizer,rewardmodel,rewardmodel_tokenizer,train_dataloader,token_false_id,token_true_id,max_new_tokens))
TestR.append(evaluateTrainedLLM(llm,llm_tokenizer,rewardmodel,rewardmodel_tokenizer,test_dataloader,token_false_id,token_true_id,max_new_tokens))
drawReward("LLM_GRPO", TrainR, TestR)
best_testr=TestR[0]
global_step = 0
for epoch in tqdm(range(n_epoch)):
    epoch_loss = 0.0

    for batch in tqdm(train_dataloader):
        B = batch["input_ids"].size(0)

        with torch.no_grad():
            answers_text, answers_token_ids = sample_answers(
                llm, batch["input_ids"], batch["attention_mask"],
                K=K, temperature=temperature, max_new_tokens=max_new_tokens
            )

        # 2) 奖励（每个样本内的 K 个回答）
        #    r_list: List[Tensor[K]]，每个元素对应一个样本组
        r_list = []
        for i in range(B):
            r = getReward(
                rewardmodel, rewardmodel_tokenizer,
                batch["query"][i], answers_text[i],
                token_false_id, token_true_id
            )  # 假设返回 shape=(K,) 的 Tensor（在 CPU 或 GPU）
            r = r.to(device).float()
            r_list.append(r)

        # 3) 组内标准化优势（A = (r - mean)/std）
        R = torch.stack(r_list, dim=0).to(torch.float32)
        var, mean = torch.var_mean(R, dim=1, unbiased=False, keepdim=True)
        std = (var + 1e-8).sqrt()
        A_flat = ((R - mean) / std).reshape(-1).to(r_list[0].dtype)
        del r_list

        # 4) 构建拼接 batch（N = B*K），用于计算当前策略/参考策略的 logprob
        concat_input_ids, concat_attention, prompt_lens, ans_lens = build_concat_batch(
            batch["input_ids"], batch["attention_mask"], answers_token_ids,
            tokenizer=llm_tokenizer, max_len=llm_max_length + max_new_tokens
        )
        concat_input_ids = concat_input_ids.to(device, non_blocking=True)
        concat_attention = concat_attention.to(device, non_blocking=True)

        # 5) 计算 logprob（当前策略 & 参考策略；对回答部分取平均）
        logp_pi  = sequence_logprobs(llm, concat_input_ids, concat_attention, prompt_lens)
        with torch.no_grad():
            logp_ref = sequence_logprobs(reference_llm, concat_input_ids, concat_attention, prompt_lens)


        # 6) GRPO 损失： -A * logp_pi  +  kl_coef * (logp_pi - logp_ref)
        #    其中 KL 项使用 token-level 平均 log 比率的近似
        loss_main = -(A_flat.detach() * logp_pi).mean()
        loss_kl   = kl_coef * (logp_pi - logp_ref).mean()
        loss = loss_main + loss_kl

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(llm.parameters(), grad_clip)
        optimizer.step()

        epoch_loss += loss.item()
        
        global_step += 1

        del concat_input_ids, concat_attention, prompt_lens
        del logp_pi, logp_ref, A_flat
        del loss_main, loss_kl, loss

    print(f"[Epoch {epoch+1}] avg loss = {epoch_loss/len(train_dataloader):.4f}")

    trainr= evaluateTrainedLLM(llm,llm_tokenizer,rewardmodel,rewardmodel_tokenizer,train_dataloader,token_false_id,token_true_id,max_new_tokens)
    testr=evaluateTrainedLLM(llm,llm_tokenizer,rewardmodel,rewardmodel_tokenizer,test_dataloader,token_false_id,token_true_id,max_new_tokens)
    if testr>best_testr:
        best_testr=testr
        llm.save_pretrained(savePath)
    TrainR.append(trainr)
    TestR.append(testr)
    drawReward("LLM_GRPO", TrainR, TestR)

drawReward("LLM_GRPO", TrainR, TestR)
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
from scripts.tools import getReward_batched, evaluateTrainedLLM, drawReward

os.environ["TOKENIZERS_PARALLELISM"] = "false"   # 或 "true"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)   # PyTorch 2.1+
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

torch.set_float32_matmul_precision("high")   # TF32 on Ampere

lr = 1e-5
K = 4
temperature = 0.7
n_epoch = 5
max_new_tokens = 512
kl_coef = 0.02
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
rewardmodel = AutoModelForCausalLM.from_pretrained(rewardmodel_modelname, device_map="cuda:1")
rewardmodel = PeftModel.from_pretrained(rewardmodel, rewardmodel_adapter_path)
rewardmodel.eval()

rewardmodel_tokenizer = AutoTokenizer.from_pretrained(rewardmodel_modelname, padding_side="left")
if rewardmodel_tokenizer.pad_token is None:
    rewardmodel_tokenizer.pad_token = rewardmodel_tokenizer.eos_token

token_false_id = rewardmodel_tokenizer.convert_tokens_to_ids("no")
token_true_id  = rewardmodel_tokenizer.convert_tokens_to_ids("yes")


llm = AutoModelForCausalLM.from_pretrained(
    llm_modelname,
    torch_dtype=torch.bfloat16,
    device_map=device
)
llm = PeftModel.from_pretrained(llm, "weight/LLM_SFT")

for n, p in llm.named_parameters():
    p.requires_grad = ("lora" in n)

llm_tokenizer = AutoTokenizer.from_pretrained(llm_modelname)
llm_tokenizer.padding_side = "left"
if llm_tokenizer.pad_token is None:
    llm_tokenizer.pad_token = llm_tokenizer.eos_token

reference_llm = AutoModelForCausalLM.from_pretrained(
    llm_modelname,
    torch_dtype=torch.bfloat16,
    device_map=device
)
reference_llm = PeftModel.from_pretrained(reference_llm, "weight/LLM_SFT")
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
    out["input_ids"] = torch.stack([torch.tensor(x["input_ids"]) for x in batch]).to(device, non_blocking=True)
    out["attention_mask"] = torch.stack([torch.tensor(x["attention_mask"]) for x in batch]).to(device, non_blocking=True)
    out["input"] = [x["input"] for x in batch]
    out["query"] = [x["query"] for x in batch]
    return out

train_dataloader = DataLoader(tokenized_train_dataset, batch_size=rlhf_llm_batch_size, shuffle=True,  collate_fn=collate_fn)
test_dataloader  = DataLoader(tokenized_test_dataset,  batch_size=rlhf_llm_batch_size, shuffle=False, collate_fn=collate_fn)

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, llm.parameters()), lr=lr, fused=True)

@torch.no_grad()
def sample_answers(model, input_ids, attention_mask, K, temperature, max_new_tokens):
    model.eval()
    B = input_ids.size(0)

    # Repeat inputs K times
    inps = BatchEncoding({
        "input_ids": input_ids.repeat_interleave(K, dim=0),
        "attention_mask": attention_mask.repeat_interleave(K, dim=0),
    })
    gen_ids = model.generate(
        **inps,
        do_sample=True,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        pad_token_id=llm_tokenizer.pad_token_id,
        eos_token_id=llm_tokenizer.eos_token_id,
        num_return_sequences=1,   # already repeated inputs
        use_cache=True,
    )
    # Slice off prompts and regroup to B lists of length K
    answers_ids  = [[] for _ in range(B)]
    for i in range(B*K):
        b = i // K
        start = input_ids[b].shape[0]
        ans_ids = gen_ids[i][start:].tolist()
        answers_ids[b].append(ans_ids)
        # decode later in batch (avoid per-sample decode)
    model.train()
    flat_ans_ids = [ids for group in answers_ids for ids in group]
    flat_texts = llm_tokenizer.batch_decode([torch.tensor(x, dtype=torch.long, device=device) for x in flat_ans_ids],   skip_special_tokens=True)
    answers_text = [flat_texts[i*K:(i+1)*K] for i in range(B)]
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
    # mask: [N, L-1] where True for answer tokens
    Lm1 = labels.size(1)
    idx = torch.arange(Lm1, device=labels.device).unsqueeze(0)
    pl = torch.as_tensor(prompt_lens, device=labels.device).unsqueeze(1)
    mask = idx >= pl

    token_logp = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    token_logp = token_logp.masked_fill(~mask, 0.0)

    lengths = mask.sum(dim=1).clamp_min(1)
    out_logp = (token_logp.sum(dim=1) / lengths)
    del logits, log_probs, labels, idx, pl, mask, token_logp
    return out_logp, lengths

llm.train()

TrainR=[]
TestR=[]

TrainR.append(evaluateTrainedLLM(llm,llm_tokenizer,rewardmodel,rewardmodel_tokenizer,train_dataloader,token_false_id,token_true_id,max_new_tokens))
TestR.append(evaluateTrainedLLM(llm,llm_tokenizer,rewardmodel,rewardmodel_tokenizer,test_dataloader,token_false_id,token_true_id,max_new_tokens))
best_testr=TestR[0]

accumulation_steps = 4  
global_step = 0

for epoch in tqdm(range(n_epoch)):
    epoch_loss = 0.0
    accum_step = 0  

    for batch in tqdm(train_dataloader):
        B = batch["input_ids"].size(0)
        input_ids=batch["input_ids"]
        attention_mask=batch["attention_mask"]
        with torch.no_grad():
            answers_text, answers_token_ids = sample_answers(
                llm, input_ids, attention_mask,
                K=K, temperature=temperature, max_new_tokens=max_new_tokens
            )

        R = getReward_batched(
        rewardmodel, rewardmodel_tokenizer,
        batch["query"],
        answers_text, 
        token_false_id,
        token_true_id,
        ).to(torch.float32)
        R=R.to(device, non_blocking=True)
        var, mean = torch.var_mean(R, dim=1, unbiased=False, keepdim=True)
        std = (var + 1e-8).sqrt()
        A_flat = ((R - mean) / std).reshape(-1).to(R.dtype)

        concat_input_ids, concat_attention, prompt_lens, ans_lens = build_concat_batch(
            input_ids, attention_mask, answers_token_ids,
            tokenizer=llm_tokenizer, max_len=llm_max_length + max_new_tokens
        )
        del answers_token_ids, ans_lens

        logp_pi, _  = sequence_logprobs(llm, concat_input_ids, concat_attention, prompt_lens)
        with torch.no_grad():
            logp_ref, _ = sequence_logprobs(reference_llm, concat_input_ids, concat_attention, prompt_lens)
            logp_ref = logp_ref.to(device, non_blocking=True)
        loss_main = -(A_flat.detach() * logp_pi).mean() / accumulation_steps
        loss_kl   = kl_coef * (logp_pi - logp_ref).mean() / accumulation_steps
        loss = loss_main + loss_kl

        loss.backward()
        epoch_loss += loss.item() * accumulation_steps

        accum_step += 1
        if accum_step % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        del concat_input_ids, concat_attention, prompt_lens
        del logp_pi, logp_ref, A_flat
        del loss_main, loss_kl, loss

    if accum_step % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

    print(f"[Epoch {epoch+1}] avg loss = {epoch_loss/len(train_dataloader):.4f}")

    trainr= evaluateTrainedLLM(llm,llm_tokenizer,rewardmodel,rewardmodel_tokenizer,train_dataloader,token_false_id,token_true_id,max_new_tokens)
    testr=evaluateTrainedLLM(llm,llm_tokenizer,rewardmodel,rewardmodel_tokenizer,test_dataloader,token_false_id,token_true_id,max_new_tokens)
    if testr>best_testr:
        best_testr=testr
        llm.save_pretrained(savePath)
    TrainR.append(trainr)
    TestR.append(testr)

drawReward("LLM_GRPO", TrainR, TestR)
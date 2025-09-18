from modelscope import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
import torch
import pandas as pd
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import BatchEncoding
from torch.nn.utils.rnn import pad_sequence
import os
from tqdm.auto import tqdm

from scripts.settings import random_seed, llm_test_ratio, device, llm_modelname, rewardmodel_modelname, llm_max_length, llm_batch_size
from scripts.tools import getReward

lr = 1e-5
K=4
temperature=0.7
n_epoch = 2
savePath = "weight/LLM_GRPO"
os.makedirs(savePath, exist_ok=True)

df = pd.read_csv("data/LLMDataset_RLHF.csv", encoding="utf-8-sig")
def row_to_sample(row):
    query=f"<|im_start|>user\nBased on the content:{row['doc']}\nAnswer the Question:{row['query']}\n/no_think<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return {
        "input": query
    }

dataset = Dataset.from_list([row_to_sample(row) for _, row in df.iterrows()])
dataset_split = dataset.train_test_split(test_size=llm_test_ratio, seed=random_seed)
train_dataset = dataset_split["train"]
test_dataset  = dataset_split["test"]

rewardmodel_adapter_path = "weight/RewardModel_QLoRA"
rewardmodel = AutoModelForCausalLM.from_pretrained(rewardmodel_modelname).to(device)
rewardmodel = PeftModel.from_pretrained(rewardmodel, rewardmodel_adapter_path)
rewardmodel.eval()
rewardmodel_tokenizer = AutoTokenizer.from_pretrained(rewardmodel_modelname, padding_side='left')

token_false_id = rewardmodel_tokenizer.convert_tokens_to_ids("no")
token_true_id = rewardmodel_tokenizer.convert_tokens_to_ids("yes")

llm = AutoModelForCausalLM.from_pretrained(
    llm_modelname,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
llm = PeftModel.from_pretrained(
    llm, 
    "weight/LLM_SFT"
).to(device)

for name, param in llm.named_parameters():
    if "lora" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

llm_tokenizer = AutoTokenizer.from_pretrained(llm_modelname, trust_remote_code=True)
llm_tokenizer.padding_side = "left"

'''
reference_llm = AutoModelForCausalLM.from_pretrained(
    llm_modelname,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
reference_llm = PeftModel.from_pretrained(reference_llm, "weight/LLM_SFT").to(device)
reference_llm.eval()
'''

def tokenize_function(example):
    query = llm_tokenizer(example["input"], padding="max_length", truncation=True, max_length=llm_max_length, return_tensors="pt")
    features ={}
    features["input_ids"]=query["input_ids"][0]
    features["attention_mask"]=query["attention_mask"][0]
    features["query"] = example["input"]

    return features

tokenized_train_dataset = train_dataset.map(tokenize_function, remove_columns=train_dataset.column_names)
tokenized_test_dataset = test_dataset.map(tokenize_function, remove_columns=test_dataset.column_names)

def collate_fn(batch):
    batch_dict = {}

    batch_dict["input_ids"] = torch.stack([torch.tensor(item["input_ids"]) for item in batch]).to(device)
    batch_dict["attention_mask"] = torch.stack([torch.tensor(item["attention_mask"]) for item in batch]).to(device)
    batch_dict["query"]=[item["query"] for item in batch]

    return batch_dict

train_dataloader = DataLoader(
    tokenized_train_dataset,
    batch_size = llm_batch_size,
    shuffle = True,
    collate_fn=collate_fn
)
test_dataloader = DataLoader(
    tokenized_test_dataset,
    batch_size = llm_batch_size,
    shuffle = False,
    collate_fn=collate_fn
)

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, llm.parameters()), lr=lr)

def sample_answers(model, input_ids, attention_mask, K, temperature, max_new_tokens):
    """
    返回：
      answers_text: List[List[str]]   [B][K]
      answers_ids:  List[List[List[int]]]  [B][K][len_ans]
    """
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

    concat_input_ids = pad_sequence(seqs, batch_first=True, padding_value=pad_id).to(device)
    concat_attention = pad_sequence(attns, batch_first=True, padding_value=0).to(device)
    return concat_input_ids, concat_attention, prompt_lens, ans_lens

llm.save_pretrained(savePath)
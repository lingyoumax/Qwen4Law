import torch
from modelscope import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import Dataset
import pandas as pd
from tqdm.auto import tqdm
from bert_score import score
import matplotlib.pyplot as plt
import numpy as np

from settings import llm_modelname, llm_test_ratio, random_seed, device

f = pd.read_csv("LLMDataset_SFT.csv", encoding="utf-8-sig")

def row_to_sample(row):
    query=f"Based on the content:{row['doc']}\nAnswer the Question:{row['query']}\n/no_think"
    return {
        "input": query,
        "output": row["answer"]
    }

df = pd.read_csv("LLMDataset_SFT.csv", encoding="utf-8-sig")

dataset = Dataset.from_list([row_to_sample(row) for _, row in df.iterrows()])
dataset_split = dataset.train_test_split(test_size=llm_test_ratio, seed=random_seed)
test_dataset  = dataset_split["test"]

base_model = AutoModelForCausalLM.from_pretrained(
    llm_modelname,
    device_map=device,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(llm_modelname, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = PeftModel.from_pretrained(
    base_model, 
    "LLM_SFT"
).to(device)
model.eval()
Candidate = []
Answers = []

for sample in tqdm(test_dataset):
    query = sample["input"]
    candidate = sample["output"]
    messages = [
        {"role": "user", "content": query}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    answer = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    Candidate.append(candidate)
    Answers.append(answer)
p, r, f = score(Candidate, Answers, lang="zh", verbose=True, model_type="bert-base-chinese", device=device)
p = np.array(p)
r = np.array(r)
f = np.array(f)
plt.figure(figsize=(30, 10))

plt.subplot(1, 3, 1)
plt.hist(p, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(np.mean(p), color='black', linestyle='--', label=f'Mean: {np.mean(p):.3f}', lw=2)
plt.xlim(0,1)
plt.xlabel('Precision')
plt.ylabel('Count')
plt.legend(loc="upper left")

plt.subplot(1, 3, 2)
plt.hist(r, bins=30, alpha=0.7, color='salmon', edgecolor='black')
plt.axvline(np.mean(r), color='black', linestyle='--', label=f'Mean: {np.mean(r):.3f}', lw=2)
plt.xlim(0,1)
plt.xlabel('Recall')
plt.ylabel('Count')
plt.legend(loc="upper left")

plt.subplot(1, 3, 3)
plt.hist(f, bins=30, alpha=0.7, color='limegreen', edgecolor='black')
plt.axvline(np.mean(f), color='black', linestyle='--', label=f'Mean: {np.mean(f):.3f}', lw=2)
plt.xlim(0,1)
plt.xlabel('F1')
plt.ylabel('Count')
plt.legend(loc="upper left")

plt.savefig("Figs/evalLLM_SFT.svg")
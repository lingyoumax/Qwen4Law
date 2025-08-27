import pandas as pd
from modelscope import AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
import os

from scripts.settings import llm_modelname

max_length=100000

df = pd.read_csv("data/LLMDataset_SFT.csv", encoding="utf-8-sig")

tokenizer = AutoTokenizer.from_pretrained(llm_modelname)

def format_messages(question, reference, answer):
    query=f"Based on the content:{reference}\nAnswer the Question:{question}\n/no_think"
    return [{"role": "user", "content": query}, {"role": "assistant", "content": answer}]

messages_token_len = []

for _, d in df.iterrows():
    messages= format_messages(d['query'], d["doc"],d['answer'])
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
        return_tensors=None
    )
    messages_token_len.append(len(prompt_ids))

messages_mean = np.mean(messages_token_len)
messages_p99 = np.percentile(messages_token_len, 99)

plt.figure(figsize=(10, 10))  
plt.hist(messages_token_len, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(messages_mean, color='red', linestyle='--', label=f'Mean: {messages_mean:.1f}')
plt.axvline(messages_p99, color='orange', linestyle='--', label=f'P99: {messages_p99:.1f}')
plt.title('Messages Token Length')
plt.xlabel('Token Length')
plt.ylabel('Count')
plt.legend()

os.makedirs("figs", exist_ok=True)
plt.savefig('figs/evalLLMTokenLength.jpg')
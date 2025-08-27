import pandas as pd
from modelscope import AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
import os

from scripts.settings import num_negative_docs, reranker_modelname

max_length=100000

df = pd.read_csv("data/RetrieverDataset_cleaned.csv", encoding="utf-8-sig")

tokenizer = AutoTokenizer.from_pretrained( reranker_modelname, padding_side='left')

def format_instruction(query, doc):
    output = "<Query>: {query}\n<Document>: {doc}".format(query=query, doc=doc)
    return output

pair_token_len = []

prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

prefix_token_len = len(prefix_tokens)
suffix_token_len = len(suffix_tokens)

for _, d in df.iterrows():
    pair = format_instruction(d['query'], d["positive_doc"])
    pair_token_len.append(prefix_token_len + suffix_token_len + len(tokenizer(pair, padding=True, truncation=True, max_length=max_length, return_tensors="pt")["input_ids"][0]))

    for i in range(num_negative_docs):
        pair = format_instruction(d['query'], d[f"negative_doc{i}"])
        pair_token_len.append(prefix_token_len + suffix_token_len + len(tokenizer(pair, padding=True, truncation=True, max_length=max_length, return_tensors="pt")["input_ids"][0]))


pair_mean = np.mean(pair_token_len)
pair_p99 = np.percentile(pair_token_len, 99)

plt.figure(figsize=(10, 10))  
plt.hist(pair_token_len, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(pair_mean, color='red', linestyle='--', label=f'Mean: {pair_mean:.1f}')
plt.axvline(pair_p99, color='orange', linestyle='--', label=f'P99: {pair_p99:.1f}')
plt.title('Pair Token Length')
plt.xlabel('Token Length')
plt.ylabel('Count')
plt.legend()


os.makedirs("figs", exist_ok=True)
plt.savefig('figs/evalRerankerTokenLength.jpg')
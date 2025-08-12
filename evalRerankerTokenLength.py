import pandas as pd
from modelscope import AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
import os

from settings import num_negative_docs, reranker_modelname

max_length=100000

df = pd.read_csv("RetrieverDataset_selfinstruct_cleaned.csv", encoding="utf-8-sig")

tokenizer = AutoTokenizer.from_pretrained( reranker_modelname, padding_side='left')

def format_instruction(query, doc):
    output = "<Query>: {query}\n<Document>: {doc}".format(query=query, doc=doc)
    return output

pair_token_len = []

for _, d in df.iterrows():
    pair = format_instruction(d['query'], d["positive_doc"])
    pair_token_len.append(len(tokenizer(pair, padding=True, truncation=True, max_length=max_length, return_tensors="pt")["input_ids"][0]))

    for i in range(num_negative_docs):
        pair = format_instruction(d['query'], d[f"negative_doc{i}"])
        pair_token_len.append(len(tokenizer(pair, padding=True, truncation=True, max_length=max_length, return_tensors="pt")["input_ids"][0]))


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


os.makedirs("Figs", exist_ok=True)
plt.savefig('Figs/evalRerankerTokenLength.jpg')
import pandas as pd
from modelscope import AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
import os

from scripts.settings import num_negative_docs, embedding_modelname

max_length=100000

df = pd.read_csv("data/RetrieverDataset_cleaned.csv", encoding="utf-8-sig")

tokenizer = AutoTokenizer.from_pretrained(embedding_modelname, padding_side='left')

query_token_len= [len(tokenizer(d["query"], padding=True, truncation=True, max_length=max_length, return_tensors="pt")["input_ids"][0]) for _, d in df.iterrows()]

postive_token_len= [len(tokenizer(d["positive_doc"], padding=True, truncation=True, max_length=max_length, return_tensors="pt")["input_ids"][0]) for _, d in df.iterrows()]

negative_token_len= [len(tokenizer(d[f"negative_doc{i}"], padding=True, truncation=True, max_length=max_length, return_tensors="pt")["input_ids"][0]) for _, d in df.iterrows() for i in range(num_negative_docs)]

query_mean = np.mean(query_token_len)
query_p99 = np.percentile(query_token_len, 99)

positive_mean = np.mean(postive_token_len)
positive_p99 = np.percentile(postive_token_len, 99)

negative_mean = np.mean(negative_token_len)
negative_p99 = np.percentile(negative_token_len, 99)


plt.figure(figsize=(30, 10))  
plt.subplot(1, 3, 1)
plt.hist(query_token_len, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(query_mean, color='red', linestyle='--', label=f'Mean: {query_mean:.1f}')
plt.axvline(query_p99, color='orange', linestyle='--', label=f'P99: {query_p99:.1f}')
plt.title('Query Token Length')
plt.xlabel('Token Length')
plt.ylabel('Count')
plt.legend()

plt.subplot(1, 3, 2)
plt.hist(postive_token_len, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
plt.axvline(positive_mean, color='red', linestyle='--', label=f'Mean: {positive_mean:.1f}')
plt.axvline(positive_p99, color='orange', linestyle='--', label=f'P99: {positive_p99:.1f}')
plt.title('Positive Doc Token Length')
plt.xlabel('Token Length')
plt.ylabel('Count')
plt.legend()

plt.subplot(1, 3, 3)
plt.hist(negative_token_len, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
plt.axvline(negative_mean, color='red', linestyle='--', label=f'Mean: {negative_mean:.1f}')
plt.axvline(negative_p99, color='orange', linestyle='--', label=f'P99: {negative_p99:.1f}')
plt.title('Negative Doc Token Length')
plt.xlabel('Token Length')
plt.ylabel('Count')
plt.legend()

plt.tight_layout()

os.makedirs("figs", exist_ok=True)
plt.savefig('figs/evalEmbeddingTokenLength.jpg')
import pandas as pd
from modelscope import AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
import os

from settings import num_negative_docs, retriever_modelname

max_length=100000

df = pd.read_csv("RetrieverDataset_selfinstruct_cleaned.csv", encoding="utf-8-sig")

tokenizer = AutoTokenizer.from_pretrained('RetrieverTokenizer', padding_side='left')

query_token_len= [len(tokenizer(d["query"], padding=True, truncation=True, max_length=max_length, return_tensors="pt")["input_ids"][0]) for _, d in df.iterrows()]

postive_token_len= [len(tokenizer(d["positive_doc"], padding=True, truncation=True, max_length=max_length, return_tensors="pt")["input_ids"][0]) for _, d in df.iterrows()]

negative_token_len= [len(tokenizer(d[f"negative_doc{i}"], padding=True, truncation=True, max_length=max_length, return_tensors="pt")["input_ids"][0]) for _, d in df.iterrows() for i in range(num_negative_docs)]

query_mean = np.mean(query_token_len)
query_p99 = np.percentile(query_token_len, 99)

positive_mean = np.mean(postive_token_len)
positive_p99 = np.percentile(postive_token_len, 99)

negative_mean = np.mean(negative_token_len)
negative_p99 = np.percentile(negative_token_len, 99)

tokenizer_qwen = AutoTokenizer.from_pretrained(retriever_modelname, padding_side='left')

query_token_len_qwen= [len(tokenizer_qwen(d["query"], padding=True, truncation=True, max_length=max_length, return_tensors="pt")["input_ids"][0]) for _, d in df.iterrows()]

postive_token_len_qwen= [len(tokenizer_qwen(d["positive_doc"], padding=True, truncation=True, max_length=max_length, return_tensors="pt")["input_ids"][0]) for _, d in df.iterrows()]

negative_token_len_qwen= [len(tokenizer_qwen(d[f"negative_doc{i}"], padding=True, truncation=True, max_length=max_length, return_tensors="pt")["input_ids"][0]) for _, d in df.iterrows() for i in range(num_negative_docs)]

query_mean_qwen = np.mean(query_token_len_qwen)
query_p99_qwen = np.percentile(query_token_len_qwen, 99)

positive_mean_qwen = np.mean(postive_token_len_qwen)
positive_p99_qwen = np.percentile(postive_token_len_qwen, 99)

negative_mean_qwen = np.mean(negative_token_len_qwen)
negative_p99_qwen = np.percentile(negative_token_len_qwen, 99)

plt.figure(figsize=(30, 20))  
plt.subplot(2, 3, 1)
plt.hist(query_token_len, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(query_mean, color='red', linestyle='--', label=f'Mean: {query_mean:.1f}')
plt.axvline(query_p99, color='orange', linestyle='--', label=f'P99: {query_p99:.1f}')
plt.title('Query Token Length - My Tokenizer')
plt.xlabel('Token Length')
plt.ylabel('Count')
plt.legend()

plt.subplot(2, 3, 2)
plt.hist(postive_token_len, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
plt.axvline(positive_mean, color='red', linestyle='--', label=f'Mean: {positive_mean:.1f}')
plt.axvline(positive_p99, color='orange', linestyle='--', label=f'P99: {positive_p99:.1f}')
plt.title('Positive Doc Token Length - My Tokenizer')
plt.xlabel('Token Length')
plt.ylabel('Count')
plt.legend()

plt.subplot(2, 3, 3)
plt.hist(negative_token_len, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
plt.axvline(negative_mean, color='red', linestyle='--', label=f'Mean: {negative_mean:.1f}')
plt.axvline(negative_p99, color='orange', linestyle='--', label=f'P99: {negative_p99:.1f}')
plt.title('Negative Doc Token Length - My Tokenizer')
plt.xlabel('Token Length')
plt.ylabel('Count')
plt.legend()

plt.subplot(2, 3, 4)
plt.hist(query_token_len_qwen, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(query_mean_qwen, color='red', linestyle='--', label=f'Mean: {query_mean_qwen:.1f}')
plt.axvline(query_p99_qwen, color='orange', linestyle='--', label=f'P99: {query_p99_qwen:.1f}')
plt.title('Query Token Length - Pretrained Tokenizer')
plt.xlabel('Token Length')
plt.ylabel('Count')
plt.legend()

plt.subplot(2, 3, 5)
plt.hist(postive_token_len_qwen, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
plt.axvline(positive_mean_qwen, color='red', linestyle='--', label=f'Mean: {positive_mean_qwen:.1f}')
plt.axvline(positive_p99_qwen, color='orange', linestyle='--', label=f'P99: {positive_p99_qwen:.1f}')
plt.title('Positive Doc Token Length - Pretrained Tokenizer')
plt.xlabel('Token Length')
plt.ylabel('Count')
plt.legend()

plt.subplot(2, 3, 6)
plt.hist(negative_token_len_qwen, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
plt.axvline(negative_mean_qwen, color='red', linestyle='--', label=f'Mean: {negative_mean_qwen:.1f}')
plt.axvline(negative_p99_qwen, color='orange', linestyle='--', label=f'P99: {negative_p99_qwen:.1f}')
plt.title('Negative Doc Token Length - Pretrained Tokenizer')
plt.xlabel('Token Length')
plt.ylabel('Count')
plt.legend()

plt.tight_layout()

os.makedirs("Figs", exist_ok=True)
plt.savefig('Figs/evalRetrieverTokenLength.jpg')
import pandas as pd
from modelscope import AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
import os

from settings import llm_modelname

max_length=100000

df = pd.read_csv("LLMDataset.csv", encoding="utf-8-sig")

tokenizer = AutoTokenizer.from_pretrained(llm_modelname)

def format_query(question, reference):
    query = f"<Question>: {question}\n<Reference>: {reference}\n<Answer>:" 
    return query

query_token_len = []
answer_token_len = []

for _, d in df.iterrows():
    query= format_query(d['query'], d["doc"])
    answer=d['answer']
    query_token_len.append(len(tokenizer(query, padding=True, truncation=True, max_length=max_length, return_tensors="pt")["input_ids"][0]))
    answer_token_len.append(len(tokenizer(answer, padding=True, truncation=True, max_length=max_length, return_tensors="pt")["input_ids"][0]))

query_mean = np.mean(query_token_len)
query_p99 = np.percentile(query_token_len, 99)
answer_mean = np.mean(answer_token_len)
answer_p99 = np.percentile(answer_token_len, 99)

plt.figure(figsize=(20, 10))  
plt.subplot(1, 2, 1)
plt.hist(query_token_len, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(query_mean, color='red', linestyle='--', label=f'Mean: {query_mean:.1f}')
plt.axvline(query_p99, color='orange', linestyle='--', label=f'P99: {query_p99:.1f}')
plt.title('Query Token Length')
plt.xlabel('Token Length')
plt.ylabel('Count')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(answer_token_len, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(answer_mean, color='red', linestyle='--', label=f'Mean: {answer_mean:.1f}')
plt.axvline(answer_p99, color='orange', linestyle='--', label=f'P99: {answer_p99:.1f}')
plt.title('Answer Token Length')
plt.xlabel('Token Length')
plt.ylabel('Count')
plt.legend()

os.makedirs("Figs", exist_ok=True)
plt.savefig('Figs/evalLLMTokenLength.jpg')
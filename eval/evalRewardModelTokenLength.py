import pandas as pd
from modelscope import AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
import os

from scripts.settings import reward_modelname

max_length=100000

df = pd.read_csv("data/LLMDataset_RLHF.csv", encoding="utf-8-sig")

tokenizer = AutoTokenizer.from_pretrained(reward_modelname, padding_side='left')

def format_instruction(instruction, query, answer):
    output = "<Instruct>: {instruction}\n<Query>: {query}\n<Answer>: {answer}".format(instruction=instruction,query=query, answer=answer)
    return output

task = 'Given a legal question, please answer it.'
prefix = "<|im_start|>system\nEvaluate the given answer based on the question, and comprehensively assess whether it is a good answer from the perspectives of accuracy, completeness, rigor, usefulness, and natural fluency. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

good_answer_token_len= [len(tokenizer(prefix+format_instruction(task,d["query"],d["answer_good"]), padding=True, truncation=True, max_length=max_length, return_tensors="pt")["input_ids"][0]) for _, d in df.iterrows()]

bad_answer_token_len= [len(tokenizer(prefix+format_instruction(task,d["query"],d["answer_bad"]), padding=True, truncation=True, max_length=max_length, return_tensors="pt")["input_ids"][0]) for _, d in df.iterrows()]


good_answer_mean = np.mean(good_answer_token_len)
good_answer_p99 = np.percentile(good_answer_token_len, 99)

bad_answer_mean = np.mean(bad_answer_token_len)
bad_answer_p99 = np.percentile(bad_answer_token_len, 99)

plt.figure(figsize=(20, 10))  
plt.subplot(1, 2, 1)
plt.hist(good_answer_token_len, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(good_answer_mean, color='red', linestyle='--', label=f'Mean: {good_answer_mean:.1f}')
plt.axvline(good_answer_p99, color='orange', linestyle='--', label=f'P99: {good_answer_p99:.1f}')
plt.title('Token Length Distribution for Queries with Good Answers')
plt.xlabel('Token Length')
plt.ylabel('Count')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(bad_answer_token_len, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(bad_answer_mean, color='red', linestyle='--', label=f'Mean: {bad_answer_mean:.1f}')
plt.axvline(bad_answer_p99, color='orange', linestyle='--', label=f'P99: {bad_answer_p99:.1f}')
plt.title('Token Length Distribution for Queries with Bad Answers')
plt.xlabel('Token Length')
plt.ylabel('Count')
plt.legend()

plt.tight_layout()

os.makedirs("figs", exist_ok=True)
plt.savefig('figs/evalRewardModelTokenLength.jpg')
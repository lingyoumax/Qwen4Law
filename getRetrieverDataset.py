import requests
import pandas as pd
import random
from tqdm import tqdm
import json
import re
from tools import getFiles

def clean_text(text):
    # 删除<think>标签及其内容 
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # 删除```json和```标记
    text = re.sub(r'```json|```', '', text)
    
    # 删除两端的空白字符
    text = text.strip()
    
    return text

def generate_with_ollama(prompt, model="qwen3:32b"):
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(url, json=payload)
    data = response.json()

    return data["response"]

num_negative_docs=10#在LLM的prompt中涉及的候选负例个数
data=[]

def getPrompt(postive_doc, negative_docs):
    prompt=f"""
请你根据正例文档生成一个用户查询(query)，这个查询能准确检索到该文档内容，对于一些关键词，可以使用语义相近的词进行替代，以实现语义检索的效果。同时需要注意，生成的用户查询不应该能查询到负例文档。
输出中只需要包含生成的用户查询，不需要其它任何内容。

【正例文档】
{postive_doc}

【负例文档列表】
"""
    for i in range(len(negative_docs)):
        prompt = prompt + f"{i+1}. {negative_docs[i]}\n"
    
    return prompt

columns_to_join = ['编', '分编', '章', '节', '内容']

def row2doc(row):
    return ",".join([str(row[col]) for col in columns_to_join if pd.notnull(row[col])])

df=pd.read_csv("Laws_SelectedKMeans.csv")
for i in tqdm(range(0,df.shape[0])):
    row = df.iloc[i]
    postive_doc = row2doc(row)
    for j in range(1):
        candidate_indices = [x for x in range(df.shape[0]) if x != i]
        nums = random.sample(candidate_indices, num_negative_docs)

        negative_docs=[]
        for k in nums:
            negative_docs.append(row2doc(df.iloc[k]))
        prompt=getPrompt(postive_doc, negative_docs)
        result = generate_with_ollama(prompt)
        try:
            result=clean_text(result)
            data.append([result, postive_doc, docs for docs in negative_docs])
        except Exception as e:
            print(e)
            print(result)
            print()

columns = ["query", "postive_doc", f"negative_doc{i}" for i in range(num_negative_docs)]
RetrieverData_selfinstruct = pd.DataFrame(data, columns=columns)
RetrieverData_selfinstruct.to_csv("RetrieverDataset_selfinstruct.csv", index=False, encoding="utf-8-sig")
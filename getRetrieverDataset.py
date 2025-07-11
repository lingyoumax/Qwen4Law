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
请你根据正例文档生成一个用户查询(query)，这个查询能准确检索到该文档内容。  
然后，从候选文档库里任选一篇与该查询无关或弱相关的文档，作为负例文档。
返回时必须使用指定的返回格式返回。

【正例文档】
{postive_doc}

【候选文档列表】
"""
    for i in range(len(negative_docs)):
        prompt = prompt + f"{i+1}. {negative_docs[i]}\n"
    
    prompt = prompt + """
【返回格式】
{"query":"用户查询",
"negative_doc":"你选中的负例文档内容"
}
"""
    return prompt
columns_to_join = ['编', '分编', '章', '节', '内容']

def row2doc(row):
    return ",".join([str(row[col]) for col in columns_to_join if pd.notnull(row[col])])

directory='laws'
fileend='.csv'
csv_files=[f"{directory}/{f}{fileend}" for f in getFiles(directory, fileend)]
dfs=[]
for file in csv_files:
    dfs.append(pd.read_csv(file))

df = pd.concat(dfs, ignore_index=True)
for i in tqdm(range(df.shape[0])):
    row = df.iloc[i]
    postive_doc = row2doc(row)
    for j in range(3):
        nums = random.sample(range(0, df.shape[0]), num_negative_docs)
        negative_docs=[]
        for k in nums:
            negative_docs.append(row2doc(df.iloc[k]))
        prompt=getPrompt(postive_doc, negative_docs)
        result = generate_with_ollama(prompt)
        try:
            result_json=json.loads(clean_text(result))
            data.append([result_json["query"], postive_doc, result_json["negative_doc"]])
        except Exception as e:
            print(e)
            print(result)
            print()

RetrieverData_selfinstruct = pd.DataFrame(data, columns=["query", "postive_doc", "negative_doc"])
RetrieverData_selfinstruct.to_csv("RetrieverDataset_selfinstruct.csv", index=False, encoding="utf-8-sig")
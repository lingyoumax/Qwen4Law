import requests
import pandas as pd
import random
from tqdm import tqdm
import re

def clean_text(text):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    text = re.sub(r'```json|```', '', text)
    
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

num_negative_docs=10
data=[]

def getPrompt(postive_doc, negative_docs):
    prompt=f"""
你的任务是根据正例文档生成一个用户查询，该查询能准确检索到正例文档内容，同时使用语义相近的词替代关键词以实现语义检索效果，并且该查询不能检索到负例文档。输出中仅包含生成的用户查询，不包含其他内容。
首先，请仔细阅读以下正例文档：
<正例文档>
{postive_doc}
</正例文档>
接着，请阅读以下负例文档列表：
<负例文档列表>
"""
    for i in range(len(negative_docs)):
        prompt = prompt + f"{i+1}. {negative_docs[i]}\n"
    
    prompt = prompt +"""\n
</负例文档列表>
在生成用户查询时，请遵循以下规则：
1. 确保查询能准确检索到正例文档内容。
2. 对于正例文档中的一些关键词，使用语义相近的词进行替代。
3. 保证生成的用户查询不会检索到负例文档中的内容。
4. 输出中只需要包含查询内容，不需要其它类似“用户查询：”的开头
5. 生成的查询需要模仿人类查询语气和用词
"""
    return prompt

columns_to_join = ['编', '分编', '章', '节', '内容']

def row2doc(row):
    return ",".join([str(row[col]) for col in columns_to_join if pd.notnull(row[col])])

df=pd.read_csv("Laws_SelectedKMeans.csv")

for i in tqdm(range(df.shape[0])):
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
            d=[result, postive_doc]
            d.extend(negative_docs)
            data.append(d)
        except Exception as e:
            print(e)
            print(result)
            print()

columns = ["query", "postive_doc"]
columns.extend([f"negative_doc{i}" for i in range(num_negative_docs)])

RetrieverData_selfinstruct = pd.DataFrame(data, columns=columns)
RetrieverData_selfinstruct.to_csv("RetrieverDataset_selfinstruct.csv", index=False, encoding="utf-8-sig")
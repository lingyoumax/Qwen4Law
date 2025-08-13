from openai import OpenAI
import config
import pandas as pd
import random
from tqdm import tqdm
import re
from settings import num_negative_docs
client = OpenAI(
    api_key=config.api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def generate_with_qwen(prompt, model="qwen3-235b-a22b-instruct-2507"):
    messages = [{"role": "user", "content": prompt}]
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        extra_body={"enable_thinking": False}
    )

    return completion.choices[0].message.content

def clean_text(text):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    text = re.sub(r'```json|```', '', text)
    
    text = text.strip()
    
    lines = text.splitlines()
    last_line = lines[-1].strip() if lines else ''

    if last_line.startswith('"') and last_line.endswith('"'):
        last_line = last_line[1:-1]
    elif last_line.startswith('“') and last_line.endswith('”'):
        last_line = last_line[1:-1]
    
    return last_line

def getPrompt(query, doc):
    prompt=f"""
你的任务是根据用户的提问和相关文档回答问题，该回答应能准确对应到文档内容，并避免引入不想关内容。请生成符合人类自然搜索习惯的回答。

# 文档说明
- 为法律条文/政府规章类文本，具有以下特点：
  - 包含文件名、章节号、编号、条号等定位信息
  - 表述严谨，多使用"应当""不得""处罚"等术语
  - 可能涉及数据标准（如罚款金额、水位高度等）

# 文档
首先，请仔细阅读以下文档：
<文档内容>
{doc}
</文档内容>

# 用户提问
<用户提问>
{query}
</用户提问>

# 生成要求
在生成回答时，请遵循以下规则：
1. 必须确保回答能准确覆盖文档的核心内容。
2. 输出中只包含回答，不需要前缀、解释或格式说明。
3. 模拟真实用户的自然语言习惯，使回答自然流畅。
4. 如果有多个自然表达方式可以作为回答，请选择最常见或最自然的一种。
5. 生成的回答必须要紧密联系文档内容。
6. 禁止引入与文档不想关的背景信息或内容。
"""
    return prompt

data=[]
df=pd.read_csv("RetrieverDataset_selfinstruct_cleaned.csv").head(5)

for i in tqdm(range(df.shape[0])):
    row = df.iloc[i]
    query = row["query"]
    doc = row["positive_doc"]
    prompt=getPrompt(query, doc)
    for j in range(1):
        result = generate_with_qwen(prompt)
        try:
            result=clean_text(result)
            d=[query, doc, result]
            data.append(d)
        except Exception as e:
            print(e)
            print(result)
            print()
    


columns = ["query", "doc", "answer"]

RetrieverData_selfinstruct = pd.DataFrame(data, columns=columns)
RetrieverData_selfinstruct.to_csv("LLMDataset.csv", index=False, encoding="utf-8-sig")
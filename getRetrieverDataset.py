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


data=[]

def getPrompt(positive_doc, negative_docs):
    prompt=f"""
你的任务是根据正例文档生成一个用户查询，该查询应能准确检索到正例文档内容，并避免检索到负例文档。请生成符合人类自然搜索习惯的语义查询。

# 文档说明
- 正例和负例均为法律条文/政府规章类文本，具有以下特点：
  - 可能包含文件名、章节号、编号、条号等定位信息
  - 表述严谨，多使用"应当""不得""处罚"等术语
  - 可能涉及数据标准（如罚款金额、水位高度等）

# 正例文档  
首先，请仔细阅读以下正例文档：
<正例文档>
{positive_doc}
</正例文档>

# 需避开的负例文档
接着，请阅读以下负例文档列表：
<负例文档列表>
"""
    for i in range(len(negative_docs)):
        prompt = prompt + f"{i+1}. {negative_docs[i]}\n"
    
    prompt = prompt +"""\n
</负例文档列表>

# 生成要求
在生成用户查询时，请遵循以下规则：
1. 必须确保查询能准确覆盖正例文档的核心内容。
2. 使用自然语言表达方式替换原始关键词，使其更贴近语义检索场景。
3. 查询语句必须避免匹配到负例文档的任何核心内容。
4. 输出中只包含查询内容，不需要前缀、解释或格式说明。
5. 模拟真实用户的搜索语言习惯，使查询自然流畅。
6. 注意比较正负文档的核心差异，避免语义混淆。
7. 如果有多个自然表达方式可以实现查询目的，请选择最常见或最自然的一种。
8. 查询必须属于以下类型之一（请根据文档内容选择最合适的类型）：
   - **事实型查询**（查询具体数据、标准、法律条款）：  
     - 示例：“抚仙湖的最高运行水位是多少？”  
   - **操作型查询**（询问如何做某事或流程）：  
     - 示例：“交通事故未移车导致堵塞会怎么处罚？”  
   - **定义型查询**（询问某个概念或法律条文的含义）：  
     - 示例：“现役军官和预备役军官的军衔有什么区别？”  
   - **政策/法规依据查询**（询问某规定的法律来源或适用范围）：  
     - 示例：“贵州省人大如何监督国有资产管理？” 
"""
    return prompt

def row2doc(row):
    return ",".join([str(x) for x in row if pd.notnull(x)])

df=pd.read_csv("Laws_Selected.csv")

for i in tqdm(range(df.shape[0])):
    row = df.iloc[i]
    positive_doc = row2doc(row)
    for j in range(1):
        candidate_indices = [x for x in range(df.shape[0]) if x != i]
        nums = random.sample(candidate_indices, num_negative_docs)

        negative_docs=[]
        for k in nums:
            negative_docs.append(row2doc(df.iloc[k]))
        prompt=getPrompt(positive_doc, negative_docs)
        result = generate_with_qwen(prompt)
        try:
            result=clean_text(result)
            d=[result, positive_doc]
            d.extend(negative_docs)
            data.append(d)
        except Exception as e:
            print(e)
            print(result)
            print()
    


columns = ["query", "positive_doc"]
columns.extend([f"negative_doc{i}" for i in range(num_negative_docs)])

RetrieverData_selfinstruct = pd.DataFrame(data, columns=columns)
RetrieverData_selfinstruct.to_csv("RetrieverDataset_selfinstruct.csv", index=False, encoding="utf-8-sig")
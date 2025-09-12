from openai import OpenAI
import pandas as pd
from tqdm.auto import tqdm
from peft import PeftModel
from modelscope import AutoModelForCausalLM, AutoTokenizer
import os
import torch

from scripts.settings import device, llm_modelname
from .config import API_KEY, BASE_URL

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

def generate_with_qwen(prompt, model="qwen3-235b-a22b-instruct-2507"):
    messages = [{"role": "user", "content": prompt}]
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        extra_body={"enable_thinking": True}
    )

    return completion.choices[0].message.content
def getQueryPrompt(doc):
    queryPrompt = f"""
# 角色与任务
你是一名法律领域的用户咨询模拟专家，需要基于提供的法律文档（doc），生成该文档能解答的真实用户问题（question）。这些问题将用于训练AI模型，使其能基于文档生成符合人类偏好的回答。

# 核心约束：问题必须与文档强关联
- 所有问题需能被提供的【文档内容】直接解答，不能超出文档范围（即文档中存在明确的答案依据）；
- 问题不能直接引用文档中的条款编号（如“《XX法》第5条是什么？”），需模拟用户“不知道具体条款，但有实际困惑”的场景；
- 问题需聚焦文档的核心内容（如文档重点讲“劳动合同解除的赔偿标准”，则问题需围绕该主题）。

# 文档内容（必读）
<文档内容>
{doc}
</文档内容>

# 生成要求（法律场景适配）
1. **真实场景性**：问题需模拟用户在实际工作中遇到的具体情况（如“我在试用期被公司辞退，公司说我不符合条件，这合法吗？”）；
2. **文档依赖性**：删除文档后，问题将无法被准确解答（确保问题与文档强绑定）；
3. **偏好区分度**：问题的回答需存在质量差异空间（如“是否详细说明每种情形的例外情况”“是否包含维权建议”）；
4. **术语通俗性**：用“被公司辞退”“合同解除”等用户易懂的表述，避免“用人单位单方解除权”等专业术语。

# 示例（基于上述文档）
【优质示例】
- 我在公司工作时因失误给公司造成了损失，公司能直接解雇我吗？
- 公司说经营困难要裁员，这合法吗？需要满足什么条件？
- 我同时在两家公司上班，现在公司发现了要辞退我，这合理吗？

【反面示例（避免）】
- 劳动合同的期限有几种？（文档未涉及，与文档无关）
- 《劳动合同法实施条例》第19条有哪些内容？（直接引用条款号，不符合用户视角）
- 被公司辞退后能领失业金吗？（文档未解答，超出范围）

# 任务
基于上述文档内容，生成1个符合要求的用户问题，每个问题单独成行，无需编号。
"""
    return queryPrompt

def getGoodPrompt(query, doc):
    prompt=f"""
# 角色与任务
你是一名资深法律从业者，需要基于用户的问题（Question）和提供的法律文档（Doc），生成专业、准确、实用的回答。该回答将作为“优质回答”用于模型训练，需体现法律领域的最佳实践。

# 输入信息
<用户问题>
{query}
</用户问题>

<法律文档>
{doc}
</法律文档>

# 优质回答的核心标准（必须满足）
1. **准确性（首要原则）**：
   - 严格依据文档内容，不编造法律依据（如条款号、适用条件必须与文档一致）；
   - 法律术语使用准确（如“解除劳动合同”不能简写为“辞退”，“不符合录用条件”不能扩大为“工作能力不足”）。

2. **完整性**：
   - 覆盖问题涉及的全部法律要点（如用户问“是否合法”，需同时说明“合法的条件”和“不合法的后果”）；
   - 补充文档隐含的关键信息（如文档提到“应当说明理由”，需解释“理由需书面形式，且明确具体”）。

3. **严谨性**：
   - 明确“适用边界”（如“仅适用于试用期，正式员工不适用此条款”）；
   - 对模糊表述进行限定（如“公司需‘证明’不符合录用条件，口头说无效”）。

4. **实用性**：
   - 包含用户可操作的建议（如“若公司无法证明，可向劳动仲裁委申请撤销解除决定”）；
   - 提示关键证据（如“需保存录用条件文件、辞退通知、工作记录等”）。

5. **自然流畅**：
   - 用用户易懂的语言解释专业内容（如“‘录用条件’指公司招聘时明确的岗位要求，如学历、技能等”）；
   - 结构清晰（分点说明，优先回答核心问题，再补充细节）。

# 反例警示（避免以下问题）
- 遗漏文档关键条款（如只说“可以辞退”，不提“需说明理由”）；
- 扩大适用范围（如将“试用期”扩展为“任何时期”）；
- 缺乏实操建议（只讲法律规定，不说用户该怎么做）；
- 使用模糊表述（如“可能合法”“大概需要证据”）。

# 输出要求
- 直接输出回答内容，无需前缀（如“回答：”）或格式标记；
- 长度控制在300字以内，重点突出，避免冗余。
"""
    return prompt

@torch.no_grad()
def getBadAnswer(llm, query, doc):
    question = f"Based on the content:{doc}\nAnswer the Question:{query}\n/no_think"
    messages = [
        {"role": "user", "content": question}
    ]
    text = llm_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    model_inputs = llm_tokenizer([text], return_tensors="pt").to(llm.device)
    generated_ids = llm.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    answer = llm_tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    return answer

llm = AutoModelForCausalLM.from_pretrained(
    llm_modelname,
    device_map=device,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
llm_tokenizer = AutoTokenizer.from_pretrained(llm_modelname, trust_remote_code=True)

llm = PeftModel.from_pretrained(
    llm, 
    "weight/LLM_SFT"
).to(device)
llm.eval()

data=[]
df=pd.read_csv("data/RetrieverDataset_cleaned.csv")

for i in tqdm(range(df.shape[0])):
    row = df.iloc[i]
    doc = row["positive_doc"]
    queryPrompt = getQueryPrompt(doc)
    try:
        query = generate_with_qwen(queryPrompt)
        goodPrompt=getGoodPrompt(query, doc)
        goodResult = generate_with_qwen(goodPrompt)
        badResult= getBadAnswer(llm, query, doc)
        d=[query, doc, goodResult, badResult]
        data.append(d)
    except Exception as e:
        print(i)
        print(e)
        print(goodResult)
        print(badResult)
        if goodResult and badResult:
            d=[query, doc, goodResult,badResult]
            data.append(d)
    if i%100 == 99:
        columns = ["query", "doc", "answer_good", "answer_bad"]
        RetrieverData_selfinstruct = pd.DataFrame(data, columns=columns)
        file_path = "data/LLMDataset_RLHF.csv"
        header = not os.path.exists(file_path)
        RetrieverData_selfinstruct.to_csv(file_path, mode="a", index=False, header=header, encoding="utf-8-sig")
        data = []
    
columns = ["query", "doc", "answer_good", "answer_bad"]
RetrieverData_selfinstruct = pd.DataFrame(data, columns=columns)
file_path = "data/LLMDataset_RLHF.csv"
header = not os.path.exists(file_path)
RetrieverData_selfinstruct.to_csv(file_path, mode="a", index=False, header=header, encoding="utf-8-sig")
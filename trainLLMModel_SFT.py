import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, BitsAndBytesConfig, TrainingArguments, Trainer

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from settings import llm_modelname, random_seed, llm_test_ratio, llm_batch_size, llm_max_length

# ============ 超参数 ============
lr = 1e-5
n_epoch = 10
savePath = "LLMModel_SFT"
os.makedirs(savePath, exist_ok=True)

# ============ 数据准备（instruction / input / output） ============
df = pd.read_csv("LLMDataset_SFT.csv", encoding="utf-8-sig")

def row_to_sample(row):
    query=f"Based on the content:{row['doc']}\nAnswer the Question:{row["query"]}\n/no_think"
    return {
        "input": query,
        "output": row["answer"]
    }

dataset = Dataset.from_list([row_to_sample(row) for _, row in df.iterrows()])
dataset_split = dataset.train_test_split(test_size=llm_test_ratio, seed=random_seed)
train_dataset = dataset_split["train"]
test_dataset  = dataset_split["test"]

# ============ Tokenizer ============
tokenizer = AutoTokenizer.from_pretrained(llm_modelname, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

IGNORE_INDEX = -100

# 利用 chat template 与推理对齐：
# messages = [{"role":"system","content":instruction},
#             {"role":"user","content":input},
#             {"role":"assistant","content":output}]
# 训练时 enable_thinking=False，不监督思维内容
def preprocess_batch(batch):

    # 仅包含 system+user，用于计算 prompt 长度（assistant 开始处）
    messages_prompt = [
        {"role": "user", "content": batch["input"]}
    ]
    # 完整样本，包含 assistant 的答案
    messages_full = messages_prompt + [{"role": "assistant", "content": batch["output"]}]

    # 1) prompt（不含答案），用于找分界点；加 generation prompt 以贴合推理时 assistant 起始标记
    prompt_ids = tokenizer.apply_chat_template(
        messages_prompt,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,        # 训练禁用思维模式
        return_tensors=None
    )
    # 2) full（含答案），作为模型输入
    full_ids = tokenizer.apply_chat_template(
        messages_full,
        tokenize=True,
        add_generation_prompt=False,  # 已含 assistant 内容，不需要再加生成提示
        enable_thinking=False,
        return_tensors=None
    )
    # ===== 截断策略：从左侧截断，尽量保留答案段 =====
    if len(full_ids) > llm_max_length:
        full_trunc = full_ids[-llm_max_length:]
    else:
        full_trunc = full_ids

    # 计算截断后有效的 prompt 长度（需要把被左截断的部分扣除）
    p_len_orig = len(prompt_ids)
    trunc_offset = len(full_ids)    - len(full_trunc)
    p_len_eff = max(0, p_len_orig - trunc_offset)

    # 构造 labels：prompt 段 -100，答案段监督
    labels = full_trunc.copy()
    labels[:p_len_eff] = [IGNORE_INDEX] * p_len_eff


    return {
        "input_ids": full_trunc,
        "attention_mask": [1] * len(full_trunc),
        "labels": labels
    }

tokenized_train = train_dataset.map(preprocess_batch, batched=True, remove_columns=train_dataset.column_names)
tokenized_test  = test_dataset.map(preprocess_batch,  batched=True, remove_columns=test_dataset.column_names)

# ============ 模型加载（4bit 量化 QLoRA） ============
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=getattr(torch, "bfloat16", torch.float16),
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    llm_modelname,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

# ============ LoRA 配置 ============
# Qwen3 通常有 q_proj/k_proj/v_proj/o_proj；若报错请打印命名并调整
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    inference_mode=False,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ============ 训练参数 ============
training_args = TrainingArguments(
    output_dir=savePath,
    per_device_train_batch_size=llm_batch_size,
    per_device_eval_batch_size=llm_batch_size,
    gradient_accumulation_steps=4,
    learning_rate=lr,
    num_train_epochs=n_epoch,
    logging_steps=50,
    save_strategy="epoch",
    eval_strategy="epoch",
    fp16=True, 
    report_to="tensorboard",
    optim="paged_adamw_32bit"
)

# ============ Collator & Trainer ============
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    label_pad_token_id=IGNORE_INDEX,
    pad_to_multiple_of=8,          # Tensor Core 友好
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    data_collator=data_collator,
    tokenizer=tokenizer, 
)

trainer.train()

model.save_pretrained(savePath)
import os
from datasets import Dataset
import torch
import pandas as pd
from transformers import DataCollatorForSeq2Seq, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from modelscope import AutoModelForCausalLM, AutoTokenizer

from settings import llm_modelname, device, random_seed, llm_test_ratio, llm_batch_size, llm_max_length

# ============ 超参数 ============
lr = 1e-5
n_epoch = 10
savePath = "LLMModel_SFT"

os.makedirs(savePath, exist_ok=True)

# ============ 数据准备 ============
df = pd.read_csv("LLMDataset.csv", encoding="utf-8-sig")

def row_to_sample(row):
    return {
        "question": row["query"],
        "reference": row["doc"],
        "answer": row["answer"]
    }

data = [row_to_sample(row) for _, row in df.iterrows()]
dataset = Dataset.from_list(data)

# 划分训练 / 测试集
dataset_split = dataset.train_test_split(test_size=llm_test_ratio, seed=random_seed)
train_dataset = dataset_split["train"]
test_dataset = dataset_split["test"]

# ============ Tokenizer ============
tokenizer = AutoTokenizer.from_pretrained(llm_modelname)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def preprocess_batch(batch):
    """把 question + reference 拼接作为输入，answer 作为标签"""
    inputs = [f"<Question>: {q}\n<Reference>: {r}\n<Answer>:" for q, r in zip(batch["question"], batch["reference"])]
    targets = batch["answer"]

    model_inputs = tokenizer(
        inputs,
        max_length=llm_max_length,
        truncation=True,
        padding="max_length"
    )
    labels = tokenizer(
        targets,
        max_length=llm_max_length,
        truncation=True,
        padding="max_length"
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = train_dataset.map(preprocess_batch, batched=True, remove_columns=train_dataset.column_names)
tokenized_test = test_dataset.map(preprocess_batch, batched=True, remove_columns=test_dataset.column_names)

# ============ 模型加载 (4bit 量化) ============
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    llm_modelname,
    quantization_config=bnb_config,
    device_map=device,
    trust_remote_code=True
)

model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

# ============ LoRA 配置 ============
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 常见设置，可以根据模型调整
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
)

# ============ Trainer ============
trainer = Trainer(
    model=model.to(device),
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

# ============ 开始训练 ============
trainer.train()

# ============ 保存模型 ============
model.save_pretrained(savePath)
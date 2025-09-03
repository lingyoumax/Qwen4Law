import os
import torch
import pandas as pd
from datasets import Dataset
from modelscope import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForSeq2Seq, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from scripts.settings import llm_modelname, random_seed, llm_test_ratio, llm_batch_size, llm_max_length

lr = 1e-5
n_epoch = 20
savePath = "weight/LLM_SFT"
os.makedirs(savePath, exist_ok=True)

df = pd.read_csv("data/LLMDataset_SFT.csv", encoding="utf-8-sig")

def row_to_sample(row):
    query=f"Based on the content:{row['doc']}\nAnswer the Question:{row['query']}\n/no_think"
    return {
        "input": query,
        "output": row["answer"]
    }

dataset = Dataset.from_list([row_to_sample(row) for _, row in df.iterrows()])
dataset_split = dataset.train_test_split(test_size=llm_test_ratio, seed=random_seed)
train_dataset = dataset_split["train"]
test_dataset  = dataset_split["test"]

tokenizer = AutoTokenizer.from_pretrained(llm_modelname, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

IGNORE_INDEX = -100

def preprocess_batch(batch):

    messages_prompt = [
        {"role": "user", "content": batch["input"]}
    ]
    messages_full = messages_prompt + [{"role": "assistant", "content": batch["output"]}]

    prompt_ids = tokenizer.apply_chat_template(
        messages_prompt,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors=None
    )
    full_ids = tokenizer.apply_chat_template(
        messages_full,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
        return_tensors=None
    )
    pad_len = 0
    if len(full_ids) > llm_max_length:
        full_trunc = full_ids[-llm_max_length:]
    else:
        pad_len = (llm_max_length - len(full_ids))
        full_trunc = full_ids + [tokenizer.pad_token_id] * pad_len

    p_len_orig = len(prompt_ids)
    trunc_offset = max(0, len(full_ids) - llm_max_length)
    p_len_eff = max(0, p_len_orig - trunc_offset)

    attention_mask = [1]*len(full_trunc)

    labels = full_trunc.copy()
    labels[:p_len_eff] = [IGNORE_INDEX] * p_len_eff

    if pad_len:
        attention_mask[-pad_len:] = [0] * pad_len
        labels[-pad_len:] = [IGNORE_INDEX] * pad_len


    return {
        "input_ids": full_trunc,
        "attention_mask": attention_mask,
        "labels": labels
    }

tokenized_train = train_dataset.map(preprocess_batch, remove_columns=train_dataset.column_names)
tokenized_test  = test_dataset.map(preprocess_batch, remove_columns=test_dataset.column_names)

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

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    label_pad_token_id=IGNORE_INDEX,
    pad_to_multiple_of=8, 
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
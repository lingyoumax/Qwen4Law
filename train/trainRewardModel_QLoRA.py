from modelscope import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import BatchEncoding
from datasets import Dataset
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from peft import TaskType, prepare_model_for_kbit_training, LoraConfig, get_peft_model
import torch
import os

from scripts.settings import random_seed, rewardmodel_test_ratio, device, rewardmodel_modelname, rewardmodel_max_length, rewardmodel_batch_size
from scripts.tools import evaluateRewardModel, drawLoss

lr = 1e-5
n_epoch = 20
savePath = "weight/RewardModel_QLoRA"
os.makedirs(savePath, exist_ok=True)

df = pd.read_csv("data/LLMDataset_RLHF.csv", encoding="utf-8-sig")

prefix = "<|im_start|>system\nEvaluate the given answer based on the question, and comprehensively assess whether it is a good answer from the perspectives of accuracy, completeness, rigor, usefulness, and natural fluency. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

def format_instruction(query, doc, answer):
    output = f"<Query>: Based on the content:{doc}\nAnswer the Question:{query}\n/no_think\n<Answer>: {answer}"
    return output

def row_to_sample(row):
    return {
        "good_prompt": prefix + format_instruction(row["query"],row['doc'],row["answer_good"]) + suffix,
        "bad_prompt": prefix + format_instruction(row["query"],row['doc'],row["answer_bad"]) + suffix
    }

dataset = Dataset.from_list([row_to_sample(row) for _, row in df.iterrows()])
dataset_split = dataset.train_test_split(test_size=rewardmodel_test_ratio, seed=random_seed)
train_dataset = dataset_split["train"]
test_dataset  = dataset_split["test"]

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    rewardmodel_modelname,
    quantization_config=bnb_config,
    device_map = device
)

model = prepare_model_for_kbit_training(model,gradient_checkpointing_kwargs={"use_reentrant":False})

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config).to(device)

model.train()
tokenizer = AutoTokenizer.from_pretrained(rewardmodel_modelname, padding_side='left')

def tokenize_function(example):
    good_prompt = tokenizer(example["good_prompt"], padding="max_length", truncation=True, max_length=rewardmodel_max_length, return_tensors="pt")
    bad_prompt = tokenizer(example["bad_prompt"], padding="max_length", truncation=True, max_length=rewardmodel_max_length, return_tensors="pt")

    features = {
        "good_prompt_input_ids": good_prompt["input_ids"][0],
        "good_prompt_attention_mask": good_prompt["attention_mask"][0],
        "bad_prompt_input_ids": bad_prompt["input_ids"][0],
        "bad_prompt_attention_mask": bad_prompt["attention_mask"][0]
    }
    return features

tokenized_train_dataset = train_dataset.map(tokenize_function, remove_columns=train_dataset.column_names)
tokenized_test_dataset = test_dataset.map(tokenize_function, remove_columns=test_dataset.column_names)

def collate_fn(batch):
    batch_dict = {}

    for key in batch[0]:
        value = torch.stack([torch.tensor(item[key]) for item in batch])
        batch_dict[key] = value.to(device)

    return batch_dict

train_dataloader = DataLoader(
    tokenized_train_dataset,
    batch_size = rewardmodel_batch_size,
    shuffle = True,
    collate_fn=collate_fn
)
test_dataloader = DataLoader(
    tokenized_test_dataset,
    batch_size = rewardmodel_batch_size,
    shuffle = False,
    collate_fn=collate_fn
)

token_false_id = tokenizer.convert_tokens_to_ids("no")
token_true_id = tokenizer.convert_tokens_to_ids("yes")

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

TrainLoss = []
TestLoss = []
TrainLoss.append(evaluateRewardModel(model, train_dataloader, token_false_id, token_true_id))
TestLoss.append(evaluateRewardModel(model, test_dataloader, token_false_id, token_true_id))

best_testloss=TestLoss[0]

for epoch in tqdm(range(n_epoch), desc="Training"):
    trainloss=0
    for batch in tqdm(train_dataloader):
        optimizer.zero_grad()
        good_prompt_dict=BatchEncoding({"input_ids":batch["good_prompt_input_ids"],"attention_mask":batch["good_prompt_attention_mask"]})
        good_batch_scores = model(**good_prompt_dict).logits[:, -1, :]
        good_true_vector = good_batch_scores[:, token_true_id]
        good_false_vector = good_batch_scores[:, token_false_id]
        good_s = good_true_vector - good_false_vector

        bad_prompt_dict=BatchEncoding({"input_ids":batch["bad_prompt_input_ids"],"attention_mask":batch["bad_prompt_attention_mask"]})
        bad_batch_scores = model(**bad_prompt_dict).logits[:, -1, :]
        bad_true_vector = bad_batch_scores[:, token_true_id]
        bad_false_vector = bad_batch_scores[:, token_false_id]
        bad_s = bad_true_vector - bad_false_vector
        
        good_loss = -torch.mean(torch.log(torch.sigmoid(good_s)))
        bad_loss = -torch.mean(torch.log(1-torch.sigmoid(bad_s)))
        loss=good_loss+bad_loss
        loss.backward()
        optimizer.step()
        trainloss=trainloss+loss.item()

    trainloss=trainloss/len(train_dataloader)
    testloss = evaluateRewardModel(model, test_dataloader, token_false_id, token_true_id)
    tqdm.write(f"Epoch {epoch}: Training Loss = {trainloss:.4f}, Test loss = {testloss:.4f}")
    TrainLoss.append(trainloss)
    TestLoss.append(testloss)
    if testloss < best_testloss:
        best_testloss = testloss
        model.save_pretrained(savePath)

drawLoss("RewardModel_QLoRA", TrainLoss, TestLoss)
import pandas as pd
import torch
import os
from tqdm.auto import tqdm
from transformers import BatchEncoding
from datasets import Dataset
from torch.utils.data import DataLoader
from modelscope import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import TaskType, prepare_model_for_kbit_training, LoraConfig, get_peft_model

from settings import device, reranker_modelname, reranker_max_length, num_negative_docs, reranker_batch_size, reranker_test_ratio, random_seed
from tools import evaluateTrainedRerankerModel

lr=1e-5
n_epoch=10
savePath = "RerankerModel_QLoRA"

if not os.path.exists(savePath):
    os.mkdir(savePath)

tokenizer = AutoTokenizer.from_pretrained(reranker_modelname, padding_side='left')
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(reranker_modelname, quantization_config=bnb_config,)
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

token_false_id = tokenizer.convert_tokens_to_ids("no")
token_true_id = tokenizer.convert_tokens_to_ids("yes")

prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

def format_pair(instruction, query, doc):
    if instruction is None:
        pair = "<Query>: {query}\n<Document>: {doc}".format(query=query, doc=doc)
    else:
        pair = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction,query=query, doc=doc)
    return prefix + pair + suffix

df = pd.read_csv("RetrieverDataset_cleaned.csv", encoding="utf-8-sig")

def row_to_sample(row):
    sample = {
        "positive_pair": format_pair(None, row["query"], row["positive_doc"]),
        "negative_pairs": [format_pair(None, row["query"], row[f"negative_doc{i}"]) for i in range(num_negative_docs)]
    }
    return sample

data = [row_to_sample(row) for _, row in df.iterrows()]

dataset = Dataset.from_list(data)
dataset_split = dataset.train_test_split(test_size = reranker_test_ratio, seed = random_seed)
test_dataset = dataset_split["test"]

def tokenize_function(example):
    positive_pair = tokenizer(example["positive_pair"], padding="max_length", truncation=True, max_length=reranker_max_length, return_tensors="pt")
    negative_pairs = tokenizer( example["negative_pairs"], padding="max_length", truncation=True, max_length=reranker_max_length, return_tensors="pt")

    features = {
        "positive_pair_input_ids": positive_pair["input_ids"][0],
        "positive_pair_attention_mask": positive_pair["attention_mask"][0],
    }

    for i in range(num_negative_docs):
        features[f"negative_pair_input_ids_{i}"] = negative_pairs["input_ids"][i]
        features[f"negative_pair_attention_mask_{i}"] = negative_pairs["attention_mask"][i]

    return features

tokenized_test_dataset = test_dataset.map(tokenize_function, remove_columns=test_dataset.column_names)

def collate_fn(batch):
    batch_dict = {}

    for key in batch[0]:
        value = torch.stack([torch.tensor(item[key]) for item in batch])
        batch_dict[key] = value.to(device)

    return batch_dict

test_dataloader = DataLoader(
    tokenized_test_dataset,
    batch_size = reranker_batch_size,
    shuffle = False,
    collate_fn=collate_fn
)

score = evaluateTrainedRerankerModel(model, test_dataloader, token_true_id, token_false_id)
print(score)
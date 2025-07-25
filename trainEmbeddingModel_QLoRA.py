from modelscope import AutoModel, AutoTokenizer, BitsAndBytesConfig
from transformers import BatchEncoding
from datasets import Dataset
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from peft import TaskType, prepare_model_for_kbit_training, LoraConfig, get_peft_model
import torch
import torch.nn.functional as F

from settings import num_negative_docs, device, max_length, random_seed, retriever_modelname

test_ratio=0.2
batch_size=2
lr=2e-5
n_epoch=10
temperature = 0.05

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModel.from_pretrained(
    retriever_modelname,
    quantization_config=bnb_config,
    device_map="auto"
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.FEATURE_EXTRACTION
)

model = get_peft_model(model, lora_config)

model.save_pretrained("EmbeddingModel_QLoRA")
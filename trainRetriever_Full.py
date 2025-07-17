from modelscope import AutoConfig, AutoModel, AutoTokenizer
from settings import device

config = AutoConfig.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
model = AutoModel.from_config(config).to(device)
tokenizer = AutoTokenizer.from_pretrained('RetrieverTokenizer', padding_side='left')
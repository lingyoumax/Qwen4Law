from modelscope import AutoConfig, AutoModel, AutoTokenizer

config = AutoConfig.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
model = AutoModel.from_config(config)
tokenizer = AutoTokenizer.from_pretrained('Tokenizer', padding_side='left')
from transformers import AutoConfig, AutoModel

config = AutoConfig.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
model = AutoModel.from_config(config)  
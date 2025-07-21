from modelscope import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-0.6B', padding_side='left')
model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
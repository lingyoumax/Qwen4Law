import torch
from modelscope import AutoTokenizer, AutoModelForCausalLM

from settings import device, reranker_modelname


tokenizer = AutoTokenizer.from_pretrained(reranker_modelname, padding_side='left')
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B").to(device)
token_false_id = tokenizer.convert_tokens_to_ids("no")
token_true_id = tokenizer.convert_tokens_to_ids("yes")

prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
import torch
from modelscope import AutoModel, AutoTokenizer, AutoModelForCausalLM
import pandas as pd

from scripts.settings import reward_modelname

tokenizer = AutoTokenizer.from_pretrained(reward_modelname, padding_side='left')

model = AutoModelForCausalLM.from_pretrained(reward_modelname).eval()

def format_instruction(instruction, query, answer):
    output = "<Instruct>: {instruction}\n<Query>: {query}\n<Answer>: {answer}".format(instruction=instruction,query=query, answer=answer)
    return output

prefix = "<|im_start|>system\nEvaluate the given answer based on the question, and comprehensively assess whether it is a good answer from the perspectives of accuracy, completeness, rigor, usefulness, and natural fluency. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

def process_inputs(pairs, max_length=8192):
    inputs = tokenizer(
        pairs, padding=False, truncation='longest_first',
        return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
    )
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    for key in inputs:
        inputs[key] = inputs[key].to(model.device)
    return inputs

token_false_id = tokenizer.convert_tokens_to_ids("no")
token_true_id = tokenizer.convert_tokens_to_ids("yes")

@torch.no_grad()
def compute_logits(inputs, **kwargs):
    batch_scores = model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores[:, 1].exp().tolist()
    return scores

task = 'Given a legal question, please answer it.'
df = pd.read_csv("data/LLMDataset_RLHF.csv", encoding="utf-8-sig")
queries = df["query"].to_list()
answer = df["answer_good"].to_list()
pairs = [format_instruction(task, query, doc) for query, doc in zip(queries, answer)]
inputs = process_inputs(pairs)
scores = compute_logits(inputs)

print("scores: ", scores)
import torch
from torch import Tensor
from modelscope import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from tqdm import tqdm

from settings import device, retriever_modelname

qwen_max_length=8192

tokenizer = AutoTokenizer.from_pretrained(retriever_modelname, padding_side='left')
model = AutoModel.from_pretrained(retriever_modelname)

model = model.to(device)
model.eval()

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def generate_all_embeddings(text_list, batch_size=4):
    all_embeddings = []

    for i in tqdm(range(0, len(text_list), batch_size)):
        batch_texts = text_list[i: i + batch_size]

        batch_dict = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=qwen_max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**batch_dict).last_hidden_state
            embeddings = last_token_pool(outputs, batch_dict['attention_mask'])

        embeddings_cpu = embeddings.cpu().numpy()
        all_embeddings.append(embeddings_cpu)

        del batch_dict, outputs, embeddings
        torch.cuda.empty_cache()

    final_embeddings = np.vstack(all_embeddings)
    return final_embeddings

def max_min_diverse_subset(text_list, embeddings, k=7000):
    embeddings_tensor = torch.tensor(embeddings, device=device)
    embeddings_normalized = embeddings_tensor / torch.norm(embeddings_tensor, dim=1, keepdim=True)

    N = embeddings_normalized.shape[0]
    selected_indices = []

    while True:
        first_index = torch.randint(0, N, (1,)).item()
        if text_list[first_index].strip().endswith("。"):
            break
    selected_indices.append(first_index)

    dist_to_selected = 1 - torch.mv(embeddings_normalized, embeddings_normalized[first_index])

    pbar = tqdm(total=k - 1)
    while len(selected_indices) < k:
        next_index = torch.argmax(dist_to_selected).item()

        if not text_list[next_index].strip().endswith("。"):
            dist_to_selected[next_index] = 0
            continue

        selected_indices.append(next_index)

        dist_new = 1 - torch.mv(embeddings_normalized, embeddings_normalized[next_index])
        dist_to_selected = torch.minimum(dist_to_selected, dist_new)

        pbar.update(1)
    pbar.close()

    return selected_indices

df = pd.read_csv("Laws_All.csv")
text_list = df["内容"].tolist()

embeddings=generate_all_embeddings(text_list)
np.save("Laws_Embeddings.npy", embeddings)

indices=max_min_diverse_subset(text_list, embeddings,10000)
df_selected = df.iloc[indices]
df_selected.to_csv("Laws_Selected.csv", index=False, encoding="utf-8-sig")
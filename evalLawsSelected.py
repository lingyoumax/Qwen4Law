import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from settings import device

@torch.no_grad()
def max_min_diverse_subset(text_list, embeddings_normalized, k=7000):
    
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

embeddings=np.load("Laws_Embeddings.npy")
embeddings = torch.tensor(embeddings, device=device, requires_grad=False)
embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)

nums_selected=[]
dist=[]
for n in range(1000,20001,1000):
    indices=max_min_diverse_subset(text_list, embeddings, n)
    embeddings_selected = embeddings[indices,:]
    chamfer_dists = torch.min(1 - embeddings @ embeddings_selected.T, dim=1).values
    chamfer_dist= torch.mean(chamfer_dists).item()
    nums_selected.append(n)
    dist.append(chamfer_dist)

plt.figure(figsize=(8,6))
plt.plot(nums_selected, dist, marker="o", linestyle="-", linewidth=2, markersize=6)

plt.title("Chamfer Distance vs. Number of Selected Samples", fontsize=14)
plt.xlabel("Number of Selected Samples", fontsize=12)
plt.ylabel("Chamfer Distance (All→Selected, cosine)", fontsize=12)

plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("Figs/evalLawsSelected.svg")
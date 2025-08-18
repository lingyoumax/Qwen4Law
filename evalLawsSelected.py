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

@torch.no_grad()
def min_cosine_dist(
    A: torch.Tensor,
    B: torch.Tensor,
    a_batch: int = 2048,
    b_batch: int = 2048,
    use_half: bool = False,
) -> torch.Tensor:
    """
    返回与 torch.min(1 - A @ B.T, dim=1).values 等价的 [N] 张量，
    但对 A 和 B 都分块以节省显存。假设 A、B 已做 L2 归一化。
    """
    device = A.device

    # 可选半精度
    if use_half and device.type == "cuda":
        A_work = A.half()
        B_work = B.half()
    else:
        A_work = A
        B_work = B

    N = A_work.size(0)
    mins_all = []

    for i in range(0, N, a_batch):
        Ai = A_work[i:i + a_batch]  # [a, D]
        # 当前 A 批次的最小距离，初始为 +inf
        mins_i = torch.full((Ai.size(0),), float("inf"),
                            device=device, dtype=Ai.dtype)

        # 遍历 B 的分块，逐块更新最小值
        for j in range(0, B_work.size(0), b_batch):
            Bj = B_work[j:j + b_batch]            # [b, D]
            sim = Ai @ Bj.transpose(0, 1)         # [a, b]
            dist = 1.0 - sim                      # 余弦距离
            blk_min = dist.min(dim=1).values      # [a]
            mins_i = torch.minimum(mins_i, blk_min)

            # 释放临时引用
            del Bj, sim, dist, blk_min

        mins_all.append(mins_i.to(A.dtype))
        del Ai, mins_i

    mins = torch.cat(mins_all, dim=0)  # [N]
    del mins_all
    return mins

df = pd.read_csv("Laws_All.csv")
text_list = df["内容"].tolist()

embeddings=np.load("Laws_Embeddings.npy")
embeddings = torch.tensor(embeddings, device=device, requires_grad=False)
embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)

N = embeddings.size(0)   
nums_selected=[]
dist=[]
for n in range(1000,10001,1000):
    indices=max_min_diverse_subset(text_list, embeddings, n)
    mask = torch.ones(N, dtype=torch.bool, device=embeddings.device)
    mask[indices] = False                 # 选中的设为 False
    embeddings_unselected = embeddings[mask] 
    embeddings_selected = embeddings[indices,:]
    chamfer_dist= (torch.mean(min_cosine_dist(embeddings_selected, embeddings_unselected)).item() + torch.mean(min_cosine_dist(embeddings_unselected, embeddings_selected)).item())/2
    nums_selected.append(n)
    dist.append(chamfer_dist)

plt.figure(figsize=(8,6))
plt.plot(nums_selected, dist, marker="o", linestyle="-", linewidth=2, markersize=6)

plt.title("Chamfer Distance vs. Number of Selected Samples", fontsize=14)
plt.xlabel("Number of Selected Samples", fontsize=12)
plt.ylabel("Chamfer Distance (cosine)", fontsize=12)

plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("Figs/evalLawsSelected.svg")
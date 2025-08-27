import pandas as pd
import numpy as np

from settings import num_negative_docs

df = pd.read_csv("data/RetrieverDataset.csv", encoding="utf-8-sig")

df = df.dropna(subset=['positive_doc'])

all_positive_docs = set(df['positive_doc'].dropna().unique())

for idx, row in df.iterrows():
    existing_docs = set([row['positive_doc']] + [row[f"negative_doc{i}"] for i in range(10) if pd.notna(row[f'negative_doc{i}'])])
    
    candidates = list(all_positive_docs - existing_docs)

    np.random.shuffle(candidates)

    candidate_idx = 0

    for i in range(num_negative_docs):
        col = f'negative_doc{i}'
        if pd.isna(row[col]):
            df.at[idx, col] = candidates[candidate_idx]
            candidate_idx += 1

df.to_csv("data/RetrieverDataset_cleaned.csv", index=False, encoding="utf-8-sig")
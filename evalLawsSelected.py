import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

data = np.load("Laws_Embeddings.npy")
df_all = pd.read_csv('Laws_All.csv')
df_selected = pd.read_csv('Laws_Selected.csv')

merged = df_all.merge(df_selected, how='left', indicator=True)
matched_indices = merged[merged['_merge'] == 'both'].index.tolist()

pca = PCA(n_components=2)
data_2d = pca.fit_transform(data)

labels = np.zeros(data.shape[0], dtype=int)
labels[matched_indices] = 1

plt.figure(figsize=(10, 6))

plt.scatter(data_2d[labels==0, 0], data_2d[labels==0, 1], 
            c='lightgray', s=5, alpha=0.5, label='All Laws')

plt.scatter(data_2d[labels==1, 0], data_2d[labels==1, 1], 
            c='red', marker='*', s=50, label='Selected Laws')

plt.legend()
plt.title('PCA Visualization of Laws Embeddings selected by Greedy Algorithm')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.grid(True)

os.makedirs("Figs", exist_ok=True)
plt.savefig('Figs/evalLawsSelected.jpg')
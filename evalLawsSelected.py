import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

data = np.load("Laws_Embeddings.npy")
df_all = pd.read_csv('Laws_All.csv')
df_selectedGreedy = pd.read_csv('Laws_SelectedGreedy.csv')
df_selectedKMeans = pd.read_csv('Laws_SelectedKMeans.csv')

mergedGreedy = df_all.merge(df_selectedGreedy, how='left', indicator=True)
matched_indicesGreedy = mergedGreedy[mergedGreedy['_merge'] == 'both'].index.tolist()
mergedKMeans = df_all.merge(df_selectedKMeans, how='left', indicator=True)
matched_indicesKMeans = mergedKMeans[mergedKMeans['_merge'] == 'both'].index.tolist()

pca = PCA(n_components=2)
data_2d = pca.fit_transform(data)

labelsGreedy = np.zeros(data.shape[0], dtype=int)
labelsGreedy[matched_indicesGreedy] = 1

plt.figure(figsize=(10, 6))

plt.scatter(data_2d[labelsGreedy==0, 0], data_2d[labelsGreedy==0, 1], 
            c='lightgray', s=5, alpha=0.5, label='All Laws')

plt.scatter(data_2d[labelsGreedy==1, 0], data_2d[labelsGreedy==1, 1], 
            c='red', marker='*', s=50, label='Selected Laws')

plt.legend()
plt.title('PCA Visualization of Laws Embeddings selected by Greedy Algorithm')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.grid(True)

os.makedirs("Figs", exist_ok=True)
plt.savefig('Figs/evalLawsSelectedGreedy.jpg')

plt.clf()

labelsKMeans = np.zeros(data.shape[0], dtype=int)
labelsKMeans[matched_indicesKMeans] = 1

plt.figure(figsize=(10, 6))

plt.scatter(data_2d[labelsKMeans==0, 0], data_2d[labelsKMeans==0, 1], 
            c='lightgray', s=5, alpha=0.5, label='All Laws')

plt.scatter(data_2d[labelsKMeans==1, 0], data_2d[labelsKMeans==1, 1], 
            c='red', marker='*', s=50, label='Selected Laws')

plt.legend()
plt.title('PCA Visualization of Laws Embeddings selected by K-Means Algorithm')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.grid(True)

os.makedirs("Figs", exist_ok=True)
plt.savefig('Figs/evalLawsSelectedKMeans.jpg')
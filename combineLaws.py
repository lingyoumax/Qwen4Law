import pandas as pd
from tools import getFiles

directory='laws'
fileend='.csv'
csv_files=[f"{directory}/{f}{fileend}" for f in getFiles(directory, fileend)]
dfs=[]
for file in csv_files:
    dfs.append(pd.read_csv(file))

df = pd.concat(dfs, ignore_index=True)

df.to_csv(f"Laws_All.csv", index=False, encoding="utf-8-sig")
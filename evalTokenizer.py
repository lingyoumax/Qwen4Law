import matplotlib.pyplot as plt
from tokenizers import ByteLevelBPETokenizer
from transformers import AutoTokenizer
import os

from tools import getFiles

directory='laws'
fileend='.txt'

files=[f"{directory}/{f}{fileend}" for f in getFiles(directory, fileend)]

tk = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
special_tokens = tk.all_special_tokens

x = []
y = []

for size in range(100, 2001, 100):
    x.append(size)
    tokenizer = ByteLevelBPETokenizer()
    
    tokenizer.train(files=files, vocab_size=size, min_frequency=2, special_tokens=special_tokens)
    
    total_tokens = 0
    count = 0

    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
            lines = text.split("\n")
            for line in lines:
                if line.strip():
                    encoded = tokenizer.encode(line)
                    total_tokens += len(encoded.tokens)
                    count += 1

    mean_tokens = total_tokens / count if count > 0 else 0
    y.append(mean_tokens)

plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.xlabel('Vocab Size')
plt.ylabel('Mean # of Tokens per Line')
plt.title('Effect of Vocab Size on Tokenization (Qwen3 Special Tokens)')
plt.grid(True)

os.makedirs("Figs", exist_ok=True)
plt.savefig('Figs/evalTokenizer.jpg')
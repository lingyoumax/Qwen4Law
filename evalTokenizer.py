import matplotlib.pyplot as plt
from tokenizers import ByteLevelBPETokenizer
import os

files=["中国民法典.txt"]
x=[]
y=[]
for size in range(100,5001,100):
    x.append(size)
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=files, vocab_size=size, min_frequency=2, special_tokens=["<s>", "<pad>", "</s>", "<unk>"])

    total_tokens = 0
    count=0
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
            lines = text.split("\n")
            for line in lines:
                encoded = tokenizer.encode(line)
                total_tokens += len(encoded.tokens)
                count +=1
    y.append(total_tokens/count)

plt.plot(x,y)
plt.xlabel('vocab_size')
plt.ylabel('# of Mean Token')
plt.savefig('Figs/evalTokenizer.jpg')
from modelscope import AutoTokenizer
from tools import getFiles

qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B", padding_side='left')

my_tokenizer = AutoTokenizer.from_pretrained("Tokenizer", padding_side='left')

directory='laws'
fileend='.txt'
files = [f"{directory}/{f}{fileend}" for f in getFiles(directory, fileend)]

total_tokens_qwen = 0
total_tokens_mine = 0
count = 0

for file in files:
    with open(file, "r", encoding="utf-8") as f:
        text = f.read()
        lines = text.split("\n")
        for line in lines:
            if line.strip():  # 跳过空行
                # 用 Qwen tokenizer 编码
                encoded_qwen = qwen_tokenizer.encode(line, add_special_tokens=True)
                total_tokens_qwen += len(encoded_qwen)

                # 用自己的 tokenizer 编码
                encoded_mine = my_tokenizer.encode(line, add_special_tokens=True)
                total_tokens_mine += len(encoded_mine)

                count += 1

avg_tokens_qwen = total_tokens_qwen / count if count > 0 else 0
avg_tokens_mine = total_tokens_mine / count if count > 0 else 0

print("The average number of tokens of pretrained tokenizer:", avg_tokens_qwen)
print("The average number of tokens of my tokenizer:", avg_tokens_mine)
import pdfplumber
import re
import pandas as pd

text_all = ""
with pdfplumber.open("doc/中国民法典.pdf") as pdf:
    for page in pdf.pages:
        text_all += page.extract_text() + "\n"


text_cleaned = re.sub(r"-\s*\d+\s*-", "", text_all)

text_cleaned = "\n".join([line.strip() for line in text_cleaned.splitlines() if line.strip()])

# 示例：按"第一章" "第二章"等切
father_blocks = re.split(r"(第[一二三四五六七八九十]+章.*)", text_cleaned)

'''
chunks = []
for i in range(1, len(father_blocks), 2):
    chapter_title = father_blocks[i]
    chapter_text = father_blocks[i+1]
    
    # 每条
    articles = re.split(r"(第[0-9]+条.*?)", chapter_text)
    for j in range(1, len(articles), 2):
        article_title = articles[j]
        article_content = articles[j+1].strip()
        chunk = {
            "parent_title": chapter_title,
            "article_title": article_title,
            "content": article_content
        }
        chunks.append(chunk)

for chunk in chunks:
    # 去多余空格
    chunk["content"] = re.sub(r"\s+", " ", chunk["content"])

df = pd.DataFrame(chunks)
df.to_csv("law_chunks.csv", index=False, encoding="utf-8-sig")
'''

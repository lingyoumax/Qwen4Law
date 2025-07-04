import pdfplumber
import re
import pandas as pd

text_all = ""
with pdfplumber.open("doc/中国民法典.pdf") as pdf:
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            page_lines = page_text.strip().split("\n")
            if len(page_lines) > 0 and re.match(r"^\s*-\s*\d+\s*-\s*$", page_lines[-1]):
                page_lines = page_lines[:-1]
            cleaned_text = "\n".join(page_lines)
            text_all += cleaned_text + "\n"

lines = text_all.split("\n")

bian_re = re.compile(r"^第[零一二三四五六七八九十百千]+编\s+")
fenbian_re = re.compile(r"^第[零一二三四五六七八九十百]+分编\s+")
zhang_re = re.compile(r"^第[零一二三四五六七八九十百千]+章\s+")
jie_re = re.compile(r"^第[零一二三四五六七八九十百千]+节\s+")
tiao_re = re.compile(r"^第[零一二三四五六七八九十百千]+条\s+")

chunks = []
current_bian = "无"
current_fenbian = "无"
current_zhang = "无"
current_jie = "无"
current_text = "无"

def save_current_tiao():
    chunks.append({
        "编": current_bian,
        "分编": current_fenbian,
        "章": current_zhang,
        "节": current_jie,
        "内容": current_text
    })
savedFlag=True

for line in lines:
    if not line:
        continue
    
    if bian_re.match(line):
        if not savedFlag:
            save_current_tiao()
        savedFlag = True
        current_bian = line
        current_fenbian = "无"
        current_zhang = "无"
        current_jie = "无"
        current_text = "无"
    elif fenbian_re.match(line):
        if not savedFlag:
            save_current_tiao()
        savedFlag = True
        current_fenbian = line
        current_zhang = "无"
        current_jie = "无"
        current_text = "无"
    elif zhang_re.match(line) or line=="附 则":
        if not savedFlag:
            save_current_tiao()
        savedFlag = True
        current_zhang = line
        current_jie = "无"
        current_text = "无"
    elif jie_re.match(line):
        if not savedFlag:
            save_current_tiao()
        savedFlag = True
        current_jie = line
        current_text = "无"
    elif tiao_re.match(line):
        if not savedFlag:
            save_current_tiao()
        savedFlag=False
        current_text = line
    else:
        if line=="附 录":
            save_current_tiao()
            break
        current_text = current_text + line

df = pd.DataFrame(chunks)
df.to_csv("law_chunk.csv", index=False, encoding="utf-8-sig")
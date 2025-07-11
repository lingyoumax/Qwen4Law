import pdfplumber
import re
import pandas as pd
from tools import getFiles

def is_chinese_char_or_punct(ch):
    return (
        '\u4e00' <= ch <= '\u9fff' or  # 中文字符
        '\u3000' <= ch <= '\u303f' or  # 中文标点符号
        ch.isdigit() or                # 阿拉伯数字（0-9）
        ch == ' '                      # 空格
    )

def cleanText(text):
    filtered_text = [ch for ch in text if is_chinese_char_or_punct(ch)]
    cleaned_text = ''.join(filtered_text)
    return cleaned_text

directory='laws'
fileend='.pdf'

laws=getFiles(directory, fileend)

bian_re = re.compile(r"^第[零一二三四五六七八九十百千]+编\s+")
fenbian_re = re.compile(r"^第[零一二三四五六七八九十百]+分编\s+")
zhang_re = re.compile(r"^第[零一二三四五六七八九十百千]+章\s+")
jie_re = re.compile(r"^第[零一二三四五六七八九十百千]+节\s+")
tiao_re = re.compile(r"^第[零一二三四五六七八九十百千]+条\s+")
for law in laws:
    text_all = ""
    with pdfplumber.open(f"{directory}/{law}{fileend}") as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                page_lines = page_text.strip().split("\n")
                if len(page_lines) > 0 and re.match(r"^\s*-\s*\d+\s*-\s*$", page_lines[-1]):
                    page_lines = page_lines[:-1]
                cleaned_text = "\n".join(page_lines)
                text_all += cleaned_text + "\n"

    lines = text_all.split("\n")

    chunks = []
    saveText = ""
    current_bian = ""
    current_fenbian = ""
    current_zhang = ""
    current_jie = ""
    current_text = ""
    count = 0

    savedFlag=True

    for line in lines:
        if not line:
            continue

        if not is_chinese_char_or_punct(line[0]):
            continue

        if bian_re.match(line):
            if not savedFlag:
                chunks.append({
                "文件名":law,
                "编": current_bian,
                "分编": current_fenbian,
                "章": current_zhang,
                "节": current_jie,
                "内容": cleanText(current_text)
                })
                saveText += current_text + "\n"

            saveText += line + "\n"
            savedFlag = True
            current_bian = line
            current_fenbian = ""
            current_zhang = ""
            current_jie = ""
            current_text = ""
        elif fenbian_re.match(line):
            if not savedFlag:
                chunks.append({
                "文件名":law,
                "编": current_bian,
                "分编": current_fenbian,
                "章": current_zhang,
                "节": current_jie,
                "内容": cleanText(current_text)
                })
                saveText += cleanText(current_text) + "\n"
            saveText += line + "\n"
            savedFlag = True
            current_fenbian = line
            current_zhang = ""
            current_jie = ""
            current_text = ""
        elif zhang_re.match(line) or line=="附则":
            if not savedFlag:
                chunks.append({
                "文件名":law,
                "编": current_bian,
                "分编": current_fenbian,
                "章": current_zhang,
                "节": current_jie,
                "内容": cleanText(current_text)
                })
                saveText += cleanText(current_text) + "\n"
            saveText += line + "\n"
            savedFlag = True
            current_zhang = line
            current_jie = ""
            current_text = ""
        elif jie_re.match(line):
            if not savedFlag:
                chunks.append({
                "文件名":law,
                "编": current_bian,
                "分编": current_fenbian,
                "章": current_zhang,
                "节": current_jie,
                "内容": cleanText(current_text)
                })
                saveText += cleanText(current_text) + "\n"
            saveText += line + "\n"
            savedFlag = True
            current_jie = line
            current_text = ""
        elif tiao_re.match(line):
            if not savedFlag:
                chunks.append({
                "文件名":law,
                "编": current_bian,
                "分编": current_fenbian,
                "章": current_zhang,
                "节": current_jie,
                "内容": cleanText(current_text)
                })
                saveText += cleanText(current_text) + "\n"
            savedFlag=False
            current_text = line
        else:
            current_text = current_text + line
    chunks.append({
    "文件名":law,
    "编": current_bian,
    "分编": current_fenbian,
    "章": current_zhang,
    "节": current_jie,
    "内容": cleanText(current_text)
    })
    saveText += cleanText(current_text) + "\n"
    df = pd.DataFrame(chunks)
    df.to_csv(f"{directory}/{law}.csv", index=False, encoding="utf-8-sig")

    with open(f"{directory}/{law}.txt", "w", encoding="utf-8") as f:
        f.write(saveText)
import os

def getFiles(directory='laws', fileend='.txt'):
    txt_files = []
    for filename in os.listdir(directory):
        if filename.endswith(fileend):
            # 去除后缀
            name_without_ext = os.path.splitext(filename)[0]
            txt_files.append(name_without_ext)
    return txt_files
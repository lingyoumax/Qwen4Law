import requests
import os
from docx2pdf import convert
import subprocess

url = 'https://flk.npc.gov.cn/api/?'

foldername = "data/laws"
if not os.path.exists(foldername):
    os.mkdir(foldername)

types={"flfg":"法律", "jcfg":"监察法规", "sfjs":"司法解释", "xffl":"宪法", "xzfg":"行政法规","dfxfg":"地方性法规"}
pages={"flfg":70, "jcfg":1, "sfjs":86, "xffl":1, "xzfg":80,"dfxfg":2492}
for type in types:
    for page in range(1,pages[type]+1):
        headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        mainurl=f"https://flk.npc.gov.cn/api/?page={page}&type={type}&searchType=title%3Baccurate&sortTr=f_bbrq_s%3Bdesc&gbrqStart=&gbrqEnd=&sxrqStart=&sxrqEnd=&sort=true&size=10&_=1752046246215"
        try:
            response = requests.get(url=mainurl)
            data_json = response.json()
        except Exception as e:
            print(e)
            print(response)
            print(page)
            continue
        try:
            for index in data_json['result']['data']:
                if index["type"]!=types[type] or index["status"]!="1":
                    continue
                id = index['id']
                title = index['title']
                url = 'https://flk.npc.gov.cn/api/detail'
                data = {
                    'id':id
                }
                new_data = requests.post(url=url,data=data,headers=headers).json()
                if new_data['result']['body'][0]['type']=='WORD':
                    ind = 0
                else:
                    ind = 1

                down_load = 'https://wb.flk.npc.gov.cn'+new_data['result']['body'][ind]['path']
                name = new_data['result']['body'][ind]['path'].split('.')[-1]
                content = requests.get(url=down_load,headers=headers).content
                docx_filename = foldername + '/' + title + '.docx'
                with open(docx_filename, mode='wb') as f:
                    f.write(content)
    
                pdf_filename = foldername + '/'  + title + '.pdf'
                convert(docx_filename, pdf_filename)
                subprocess.call("taskkill /im WINWORD.EXE /f", shell=True)   
        except Exception as e:
            print(e)
            try:
                print(title)
            except:
                print()    
        print(page)
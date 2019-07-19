import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
  
def getHTMLText(url):
    try:
        r = requests.get(url, timeout=200)
        r.raise_for_status()
        return r.text
    except Exception as e:
        
        print("Get HTML Text Failed!")
        print(e)
        return 0
  
def google_translate_EtoC(to_translate, from_language="en", to_language="ch-CN"):
    #根据参数生产提交的网址
    base_url = "https://translate.google.cn/m?hl={}&sl={}&ie=UTF-8&q={}"
    url = base_url.format(to_language, from_language, to_translate)
    #获取网页
    time.sleep(0.5)
    html = getHTMLText(url)
    if html:
        soup = BeautifulSoup(html, "html.parser")
      
    #解析网页得到翻译结果   
    try:
        result = soup.find_all("div", {"class":"t0"})[0].text
    except:
#        print("Translation Failed!")
#        result = ""
        n=len(to_translate)
        result=google_translate_EtoC(to_translate[0:n-100])

          
    return result

 
def main():
    word = pd.read_csv('d://processed2news.tsv', sep='\t', encoding='utf-8', index_col=False)   
    words = pd.read_csv('d://processed1news.tsv', sep='\t', encoding='utf-8', index_col=False)
    for i in range (len(word['content'])):
        if (str(word['content'][i])=='nan'):
            content=words['content'][i]
            if (len(words['content'][i])>2001):
                content=words['content'][i][0:2000]
              
            content=content.replace('\xa0','')
            content=content.replace(' ','')
            word['content'][i]=google_translate_EtoC(content)
            print(i)
    word.to_csv('d://processed3news.tsv', sep='\t', encoding='utf-8', index=False)
 
main()
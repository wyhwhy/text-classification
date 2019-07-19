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
    word = pd.read_csv('d://processed3news.tsv', sep='\t', encoding='utf-8', index_col=False)
    word1=pd.read_csv('d://processed1news.tsv', sep='\t', encoding='utf-8', index_col=False)
    #for i in [5242,5243,5284,5448,5571,5801,5803,5825,7047,7058,12516,12558,12781]:
    for i in [509,519]:
        word['content'][i]=google_translate_EtoC(word1['content'][i])
        print(word['content'][i])
    word.to_csv('d://processed4news.tsv', sep='\t', encoding='utf-8', index=False)
    
main()

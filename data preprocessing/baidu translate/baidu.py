import hashlib
import random
import pandas as pd
import requests
 
# set baidu develop parameter
apiurl = 'http://api.fanyi.baidu.com/api/trans/vip/translate'
appid = '20190701000313590'
secretKey = 'oGgUqsrGQl3GV2Oxrd_8'
 
# 翻译内容 源语言 翻译后的语言
def translateBaidu(content, fromLang='auto', toLang='zh'):
    salt = str(random.randint(32768, 65536))
    sign = appid + content + salt + secretKey
    sign = hashlib.md5(sign.encode("utf-8")).hexdigest()
    try:
        paramas = {
            'appid': appid,
            'q': content,
            'from': fromLang,
            'to': toLang,
            'salt': salt,
            'sign': sign
        }
        response = requests.get(apiurl, paramas)
        jsonResponse = response.json()  # 获得返回的结果，结果为json格式
        dst = str(jsonResponse["trans_result"][0]["dst"])  # 取得翻译后的文本结果
        return dst
    except Exception as e:
        print(e)
def excelTrans():
    words = pd.read_csv('d://processed1news.tsv', sep='\t', encoding='utf-8', index_col=False)
    word=words.head(50)
    for i in range (len(word['content'])):
        content=word['content'][i]   
        word['content'][i]=translateBaidu(content)
        print(word['content'][i])
    word.to_csv('d://processed2news.tsv', sep='\t', encoding='utf-8', index=False)

if __name__ == '__main__':
    excelTrans()
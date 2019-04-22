# coding: UTF-8
import time
import os
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import urllib.parse


#指定した検索ワードから画像を取得する
#yahoo画像検索の圧縮画像をスクレイピングする

serchword = "ここに検索ワードを入力する"

#urlの作成
#yahoo画像検索は、pパラメータに検索ワード、bパラメータに現在の表示インデックス(1,21,41のように20件ずつ)が設定されている
url_front = "https://search.yahoo.co.jp/image/search?fr=top_ga1_sa&p="
url_middle = urllib.parse.quote(serchword)  #検索ワードをurlエンコーディング
url_back  = "&ei=UTF-8&b="
url = url_front + url_middle + url_back

#検索ワードで検索したhtmlを取得する
print("html取得開始")
headers = {'User-Agent': 'MyCrawler/1.0.0 (YOUR_EMAIL_ADDRESS)'}    #User-Agentに連絡用メールアドレスを設定
htmls = []          #html格納用リスト
for num in range(25):
    param = 1 + num * 20    #表示インデックスを取得
    r = requests.get(url + str(param),headers=headers,timeout=3.5)  #httpGet
    time.sleep(1)           #１秒停止
    htmls.append(r.text)    #htmlを取得
    print(str(num+1) + "ページ目の取得完了")
print("html取得完了")

print("画像のスクレピング開始")
#htmlから画像urlを取得
link_list = []
for html in htmls:
    soup = BeautifulSoup(html, "html.parser")
    #yahoo画像検索のhtmlでは、gridmoduleクラスのdiv要素内のimgタグのsrc属性に画像urlがある
    gridmodule_list = soup.find_all("div", attrs={"class","gridmodule"})
    for index,gridmodule in enumerate(gridmodule_list):
        img_tag = gridmodule.find('img')
        link = img_tag.attrs['src']
        link_list.append(link)

true_link_list = set(link_list)         #重複している画像urlを除外する
link_len = len(true_link_list)          #リンク数を取得
print("総リンク数は" + str(link_len))

#保存用ディレクトリを作成
print("現在時刻で保存用ディレクトリを作成")
work_dir = "/Users/kawasaki/Desktop/tmp/"
datetime = datetime.now()
dir_name = datetime.strftime("%Y%m%d%H%M%S")
os.mkdir(work_dir + dir_name)

#scrayping images
print("画像のダウンロード開始")
for index,link in enumerate(true_link_list):
    response = requests.get(link,headers=headers,timeout=3.5)   #httpGet
    time.sleep(1)   #1秒停止
    with open(work_dir + dir_name + "/" + str(index+1) + ".jpg",'wb') as f:
        f.write(response.content)   #画像を保存
    print(str(index+1) + "枚目の画像をダウンロード完了")

print("スクレイピング終了")
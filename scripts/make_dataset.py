import cv2
import numpy as np
import sys
import os
import glob
import gc
import pickle

#１クラス１フォルダに分類した画像データを１つのフォルダに集めたフォルダに対し、
#フォルダ内のデータをnumpyオブジェクトにしてpickleファイルに保存する。
#ターゲットディレクトリを引数として取る

#引数から各ディレクトリ名を取得する
arg = sys.argv
target_dir = arg[1]     #第一引数を取得：第一引数にはターゲットディレクトを入力すること
class_dir_list = os.listdir(target_dir) #ターゲットディレクトリ内のディレクトリ名を取得する
if '.DS_Store' in class_dir_list:       #macの場合、ルート権限で実行すると.Ds_Storeディレクトリが出るので削除する
    class_dir_list.remove('.DS_Store')
print(class_dir_list)

#ディレクトリ名をクラス名とし、クラス名：ファイルリストのdictオブジェクトを作成
dict_filelist = {}
for class_dir in class_dir_list:
    find_query = target_dir + '/' + class_dir + '/*.png'
    filename_list = glob.glob(find_query)
    dict_filelist[class_dir] = filename_list
for label,filenamelist in dict_filelist.items():
    print(label + ':' + str(len(filenamelist)) + 'images')

#画像データをnumpy形式にする
image_data = np.array([]).reshape(0,224,224)    #画像行列格納用
label_data = np.array([])                       #ラベル（数値）ベクトル格納用
index_to_label = {}                             #ラベル（数値）→ラベル（クラス名）に対応したdictオブジェクト格納用
for i,(label,filenamelist) in enumerate(dict_filelist.items()):
    images = []
    labels = []
    index_to_label[i] = label
    for k,filename in enumerate(filenamelist):
        img = cv2.imread(filename)      #画像をロード
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)     #グレースケール画像に変換
        #224*224の行列に変換
        resized_img = cv2.resize(gray_img,dsize=(224,224),interpolation=cv2.INTER_LINEAR)
        images.append(resized_img)  #imagesに追加
        labels.append(i)            #labelsに追加
        #データ数は多いときは、5000枚の画像ごとにimage_dataに格納した後、imagesのメモリを解放
        if(k % 5000 == 0):
            if(k != 0):
                image_data = np.append(image_data,images,axis=0)
                del images
                gc.collect()
                images = []
    image_data = np.append(image_data,images,axis=0)    #image_dataに追加
    label_data = np.append(label_data,labels,axis=0)    #label_dataに追加
    print(image_data.shape)
    print(label_data.shape)
    del images,labels           #オブジェクトを破棄
    gc.collect()                #メモリ解放

print(image_data.shape)
print(label_data.shape)

#pickleファイルに保存
print("ファイルに保存します")
with open(target_dir + '.pickle', mode='wb') as f:
    pickle.dump(image_data.astype(np.uint8), f)
    pickle.dump(label_data.astype(np.uint8),f)
    pickle.dump(index_to_label,f)

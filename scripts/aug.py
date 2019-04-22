import cv2
import sys


#データ拡張
#画像を４５度ずつ回転し、８つのファイルを出力する
#画像ファイルパスを第一引数に取る

arg = sys.argv
path = arg[1]       #画像ファイルのパスを取得
print(path)
filename = path.split('/')[-1]  #パスからファイル名だけ抽出
img = cv2.imread(path)          #画像をロード

height = img.shape[0]           #縦ピクセル数
width = img.shape[1]            #横ピクセル数
center = (int(width/2),int(height)/2)   #センター座標のタプル

#0から45度ずつ回転しながら画像を保存する
for r in range(0,315+1,45):
    angle = r
    trans = cv2.getRotationMatrix2D(center, angle , 1.0)        #回転行列を取得
    rotaed_img = cv2.warpAffine(img, trans, (width,height))     #回転した画像を取得
    write_filename = filename[0:-4] + '_r' + str(r) + '.png'    #ファイル名
    cv2.imwrite(write_filename,rotaed_img)                      #画像を保存

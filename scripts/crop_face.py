import cv2
import sys


#画像から顔を検出し、ファイルに保存する
#画像ファイルパスを第一引数に取る

args = sys.argv
path = args[1]      #ファイルパスを取得
filename = path.split('/')[-1]  #パスからファイル名だけを抽出
img = cv2.imread(path)   #画像をロード
grayscale_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)   # グレースケール画像に変換

# カスケードファイルのパス
cascade_path = "学習済みカスケードファイルのパスをここに入力"
# カスケード分類器の特徴量取得
cascade = cv2.CascadeClassifier(cascade_path)
# 顔検出、minSize 最小サイズ指定
front_face_list = cascade.detectMultiScale(grayscale_img, minSize = (30, 30))

# 検出判定
if len(front_face_list) == 0:
    print(filename + "は顔検出できませんでした")
    quit()
# 検出した顔画像をpng形式で保存
for index,(x,y,w,h) in enumerate(front_face_list):
    face_image = img[y:y+h,x:x+w]
    write_filename = filename[0:-4] + "_face_crop_" + str(index) + ".png"
    cv2.imwrite(write_filename,face_image)


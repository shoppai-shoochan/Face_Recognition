import camera
import cv2
import sys
import numpy as np
from face_detection import FaceDetection
from load_datasets import LoadDataset
import face_rec
import pickle

dir = "../resource/"  #パラメータファイル保存されているディレクトリをここに入力
cascade_file = "haarcascade_frontalface_alt.xml"
pickle_file = "aug_x8.pickle"
model_file = "alexnet_model.json"
weight_file = "alexnet_arashi_mlp_weights.h5"

cap = camera.Camera(1)              #カメラをオープン
if(not cap.CameraIsOpened()):
    print("camera is not opened")
    sys.exit()

detect = FaceDetection(dir+cascade_file)    #顔抽出用のface_detectionオブジェクトを作成
facerec = face_rec.FaceRec(dir,model_file,weight_file)    #顔認証用のface_recオブジェクトを作成

#ラベルデータを取得するため、データセットをロード
with open(dir + pickle_file,mode='rb') as f:
        image_data = pickle.load(f)         #使わない
        label_data = pickle.load(f)         #使わない
        index_to_label = pickle.load(f)     #ラベル(数値）→ラベル(文字列)にするdict型データをロード

print('camera opend')
cv2.namedWindow("faces",cv2.WINDOW_AUTOSIZE)    #顔画像表示用ウィンドウ
cv2.namedWindow("camera",cv2.WINDOW_NORMAL)     #カメラ画像表示用ウィンドウ
key = 0

#mainループ
while(key != ord("q")):         #'q'で停止
    key = cv2.waitKey(50)       #キー入力待ち(50msce)
    img = cap.ReadCaputure()    #カメラからframe画像を取得
    draw_img = img.copy()       #描画用の画像を取得
    p_faces = detect.detect(draw_img)   #顔の座標を取得
    detect.draw_rec(draw_img, p_faces)  #顔画像を赤線で描画
    cv2.imshow("camera",draw_img)       #画像表示
    if(len(p_faces)!=0):
        catface,images = detect.face_images(img, p_faces)   #顔画像を取得
        cv2.imshow("faces",catface)
        if(key == ord("n")):        #'n'押下で推論
            for image in images:
                gray_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)#グレースケール画像に変換
                predict = facerec.predict(gray_img)  #推論
                predict_label = index_to_label[predict[0]]  #推論ラベルを取得
                print("predict: " + predict_label)



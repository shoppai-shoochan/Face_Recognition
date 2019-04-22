import cv2
from keras.models import model_from_json
from keras.optimizers import SGD


class FaceRec:

    def __init__(self,dir,json_model_file,weight_file):
        print("モデル読み込み開始")
        model_json = open(dir + json_model_file,mode='r').read()
        self.model = model_from_json(model_json)
        self.model.summary()
        self.model.compile(loss="categorical_crossentropy",optimizer=SGD(),metrics=["accuracy"])
        print("モデル読み込み完了")
        print("パラメータ読み込み開始")
        self.model.load_weights(dir + weight_file)
        print("パラメータ読み込み完了")

    def predict(self,img):
        predict = self.model.predict_classes(img.reshape(1,224,224,1)).astype(int)  #推論
        return predict

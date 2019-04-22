import cv2

#画像から顔部分を抽出するクラス
class FaceDetection:
    def __init__(self,path):
        self.cascade = cv2.CascadeClassifier(path)  #カスケード分類器を取得

    #画像から顔の座標を取得
    def detect(self,img):
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)     #グレースケール画像に変換
        face_list = self.cascade.detectMultiScale(gray_img,minSize=(50,50)) #顔の座標を取得
        return face_list

    #画像の中の顔部分に赤枠をつける
    def draw_rec(self,img,face_list):
        if(not len(face_list) == 0):
            for (x,y,w,h) in face_list:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),thickness=5)

    #画像から顔画像を取得する
    def face_images(self,img,face_list):
        face_images = []
        for(x,y,w,h) in face_list:
            face_image = img[y:y+h,x:x+w]   #顔画像を取得
            scaled = cv2.resize(face_image,dsize=(224,224)) #100*100にリサイズ
            face_images.append(scaled)
        return cv2.vconcat(face_images),face_images
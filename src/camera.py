import cv2

class Camera:

    def __init__(self,camera_num=0):
        self.camera = cv2.VideoCapture(camera_num)

    def CameraIsOpened(self):
        return self.camera.isOpened()

    def ShowCaputure(self,stopKey=ord('q'),interval = 50):
        while(cv2.waitKey(interval) != stopKey):
            cv2.imshow("camera",self.ReadCaputure())

    def ReadCaputure(self):
        ret,frame = self.camera.read()
        return frame
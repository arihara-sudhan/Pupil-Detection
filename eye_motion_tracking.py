import cv2 as eye
import numpy as np
class ARI_IS_AN_EYEKILLER:
    def __init__(self):
        self.cap = eye.VideoCapture('eye1.flv')
        self.read()
    def read(self):
        while True:
            ret,frame = self.cap.read()
            if ret:
                frame = frame[369:795,537:1416]
                frame1 = eye.resize(frame,(1920,1080))
                grayframe = eye.cvtColor(frame1,eye.COLOR_BGR2GRAY)
                # Noise Removal
                grayframe = eye.GaussianBlur(grayframe,(7,7),0)
                # Let's Threshold
                ret,grayframe = eye.threshold(grayframe,3,255,eye.THRESH_BINARY_INV)
                # Contour Detection
                contours,_ = eye.findContours(grayframe,eye.RETR_TREE,eye.CHAIN_APPROX_SIMPLE)
                # Pupil Area is Bigger! Let's draw a rectangle on it only
                contours = sorted(contours,key = lambda x: eye.contourArea(x),reverse=True)
                for cnt in contours:
                    (x,y,w,h) = eye.boundingRect(cnt)
                    print(x,y,w,h)
                    eye.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
                    break
                #   eye.imshow('eye',frame1)
                eye.imshow('eye',frame1)
                if eye.waitKey(20)==27:
                    self.cap.release()
                    eye.destroyAllWindows()
ARI_IS_AN_EYEKILLER()
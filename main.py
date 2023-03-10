import cv2 
import numpy as np 
import matplotlib.pyplot as plt
from enum import Enum, auto

class FLAG(Enum):
    DEFAULT = auto(),
    COORDS = auto()


face = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")
eye = cv2.CascadeClassifier("cascades/haarcascade_eye.xml")
lbp_face = cv2.CascadeClassifier("cascades/lbpcascade_frontalface.xml")

#400y, 680y - высота
#260x, 1660x- ширина

deal =  plt.imread('images/dealwithit.png')
deal = deal[400:680, 260:1660, :3]
print(deal.shape)

cam = cv2.VideoCapture(6)
cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

cv2.namedWindow("Camera", cv2.WINDOW_KEEPRATIO)

def dealWithIt(frame, face, eyes):
    x, y, w, h, ret = getFaceRect(frame, face)
    result = frame.copy()
    if ret:
        eyes_rects = detector(frame[y:y+h, x:x+w], eyes, FLAG.COORDS)
        
        if len(eyes_rects) == 2:
            w1 = eyes_rects[0][2]
            h1 = eyes_rects[0][3]
            
            w2 = eyes_rects[1][2]
            h2 = eyes_rects[1][3]
            
            x1 = eyes_rects[0][0]
            x2 = eyes_rects[1][0]
            
            y1 = eyes_rects[0][1]
            y2 = eyes_rects[1][1]
            
            min_y = min(y1, y2) + y
            min_x = min(x1, x2) + x 
            max_x = max(x1 + w1, x2 + w2) + x
            max_y = max(y1 + h1, y2 + h2) + y
            
            new_deal = cv2.resize(deal, (max_x - min_x, max_y - min_y))

            mask = new_deal == (1, 1, 1)
            # plt.imshow(new_deal)
            # plt.waitforbuttonpress(0)
            new_deal[mask] = result[min_y:max_y, min_x:max_x, :][mask]
            result[min_y:max_y, min_x:max_x, :] = new_deal
        
    
    return result
    
def getFaceRect(img, classifier=face):
    face_rect = detector(img, classifier, FLAG.COORDS)
    
    if len(face_rect) > 0:
        x, y, w, h = face_rect[0]
        return x, y, w, h, True
    
    return 0,0,1,1, False
    
def detector(img, classifier, flag=FLAG.DEFAULT, scale=None, min_nbs=None):
    result = img.copy()
    rects = classifier.detectMultiScale(result, scaleFactor=scale, minNeighbors=min_nbs)

    match flag:
        case FLAG.DEFAULT:
            for (x, y, w, h) in rects:
                cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0))
            return result
        case FLAG.COORDS:
            return rects

def getEyesRect(img, classifier):
    pass

while cam.isOpened():
    _, frame = cam.read()   
    
    frame = dealWithIt(frame, face, eye)
    
    key = cv2.waitKey(1)
    
    if key == ord('q'):
        break
    
    cv2.imshow("Camera", frame)

cam.release()
cv2.destroyAllWindows()
import cv2
from mtcnn import MTCNN
import numpy as np
import pandas as pd
import tensorflow as tf
import sys

output = pd.DataFrame(columns=['frame num','person id','bb_xmin','bb_ymin','bb_height','bb_width','age_min','age_max','age_actual','gender'])

facedetector = MTCNN()
bodydetector = cv2.CascadeClassifier('haarcascade_fullbody.xml')


Videopath = str(sys.argv)[1]
cap = cv2.VideoCapture(Videopath)

def func(a):
    return a[0]['box'][0]

framenumber = 1

while True:
    ret, image = cap.read()

    faces = facedetector.detect_faces(image)
    if faces == []:
        framenumber += 1
        continue;
    #faces.sort(key=func)

    ID = 0

    for face in faces:
        (x, y, w, h) = face['box']
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        ID += 1
        cv2.putText(image, str(ID), (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 1)
        newrow = [{
            'frame num':framenumber,
            'person id':ID,
            'bb_xmin':x,
            'bb_ymin':y,
            'bb_height':h,
            'bb_width':w,
            'age_min':0,
            'age_max':0,
            'age_actual':0,
            'gender':0
        }]
        output = output.append(newrow)

    framenumber += 1

    cv2.imshow('img', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

output.to_csv('output.csv')
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

age_model = tf.keras.models.load_model('Age_model')
gender_model = tf.keras.models.load_model('Gender_model')


framenumber = 1

while True:
    ret, image = cap.read()

    faces = facedetector.detect_faces(image)
    if faces == []:
        cv2.imshow('img', image)
        framenumber += 1
        continue;

    ID = 0

    for face in faces:
        (x, y, w, h) = face['box']
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = image[y:y+h,x:x+w]
        ID += 1
        cv2.putText(image, str(ID), (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 1)

        predicted_age_actual = age_model.predict(face)
        predicted_gender = np.argmax(gender_model.predict(face))

        predicted_age_min = int(predicted_age_actual / 10) * 10
        predicted_age_max = predicted_age_min + 10

        newrow = [{
            'frame num':framenumber,
            'person id':ID,
            'bb_xmin':x,
            'bb_ymin':y,
            'bb_height':h,
            'bb_width':w,
            'age_min':predicted_age_min,
            'age_max':predicted_age_max,
            'age_actual':predicted_age_actual,
            'gender':predicted_gender
        }]
        output = output.append(newrow)

    framenumber += 1

    cv2.imshow('img', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

output.to_csv('output.csv')

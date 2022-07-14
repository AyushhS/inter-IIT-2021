import cv2
from mtcnn import MTCNN
import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import os
import matplotlib.pyplot as plt

output = pd.DataFrame(columns=['frame num','person id','bb_xmin','bb_ymin','bb_height','bb_width','age_min','age_max','age_actual','gender'])

# loading required models/frameworks
facedetector = MTCNN()
age_model = tf.keras.models.load_model('Age_model')
gender_model = tf.keras.models.load_model('Gender_model')

# File handling
Videopath = str(sys.argv[1])
outputfolder = []
if len(sys.argv) == 1:
    os.mkdir('output')
    outputfolder = 'output'
else:
    if os.path.isdir(sys.argv[2]) == True:
        outputfolder = sys.argv[2]
    else:
        outputfolder = os.mkdir(sys.argv[2])
if os.path.isfile(Videopath) == True:
    videos = [Videopath]
else:
    videos = [Videopath + '\\' + i for i in os.listdir(Videopath)]

print(videos)
# Video Processing
for video in videos:

    # Taking in individual videos
    cap = cv2.VideoCapture(video)
    out = cv2.VideoWriter(outputfolder + '\\' + video.split('\\')[-1].split('.')[0] + '_output.' + video.split('\\')[-1].split('.')[-1], cv2.VideoWriter_fourcc(*'MJPG'), cap.get(cv2.CAP_PROP_FPS), (int(cap.get(3)), int(cap.get(4))))

    # Processing
    framenumber = 1
    while(cap.isOpened()):
        ret, image = cap.read()

        if ret == True:
            # Detecting Faces
            faces = facedetector.detect_faces(image)
            if faces == []:
                # cv2.imshow('img', image)
                framenumber += 1
                continue;

            # Drawing boxes and detecting age and gender
            ID = 0
            image2 = image.copy()
            for face in faces:
                (x, y, w, h) = face['box']
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 4)
                face = image2[y:y+h,x:x+w]
                face = cv2.resize(face, (48,48))
                face = face.reshape(1, 48, 48, 3)
                ID += 1
                predicted_age_actual = age_model.predict(face)
                predicted_age_min = int(predicted_age_actual / 10) * 10
                predicted_age_max = predicted_age_min + 10
                gp = gender_model.predict(face)
                if (gp[0][1] > gp[0][0]):
                    predicted_gender = 'M'
                else:
                    predicted_gender = 'F'
                cv2.putText(image, str(predicted_gender) + ',' + str(predicted_age_min) + ' - ' + str(predicted_age_max), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 1)

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

            # cv2.imshow('img', image)
            out.write(image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    output.to_csv(outputfolder + '\\' + video.split('\\')[-1].split('.')[0] + '_output.csv')
import cv2
import os
import numpy as np
from mtcnn import MTCNN

folder = 'E:\coding\\test_python\inter IIT Bosch face detection (5-3-2022)\\akash\'s work\\face_recog\\train'
destination_folder = 'E:\coding\\test_python\inter IIT Bosch face detection (5-3-2022)\\akash\'s work\\face_recog\\train_faces'
paths = [folder + '\\' + i for i in os.listdir(folder)]

detector = MTCNN()
print('Done!')
for i in paths[15980:]:
    img = cv2.imread(i)
    faces = detector.detect_faces(img)
    if faces == []:
        continue
    faces = faces[0]
    (x, y, w, h) = faces['box']
    img = img[y:y+h, x:x+w]
    j = destination_folder + '\\' + i.split('\\')[-1]
    cv2.imwrite(j, img)
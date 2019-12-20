from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import csv
import math
import os

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

image = cv2.imread('datafoto_uji/s51_uji/01.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
shape = []
rects = detector(gray, 1)

id = 1
for rect in rects:
    x = rect.left()
    y = rect.top()
    w = rect.right() - rect.left()
    h = rect.bottom() - rect.top()
    cropped = gray[ y : y+h, x : x+w ]
    cropped = imutils.resize(cropped, width=500)
    id=+1

rects = detector(cropped, 1)

csvFile = "testingdata.csv"
data = []
i = 0

for (i, rect) in enumerate(rects):

    shape = predictor(cropped, rect)
    shape = face_utils.shape_to_np(shape)

    #(x, y, w, h) = face_utils.rect_to_bb(rect)
    #cv2. rectangle(cropped, (x, y), (x + w, y + h), (0, 0, 0), 2)

    #cv2.putText(gray, "Face#{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)'''

    for (x,y) in shape:
        cv2.circle(cropped, (x,y), 5, (255,0,0), -1)

    cv2.imshow('', cropped)

#right eyes 1
bucket = abs(int(shape[36][0]) - int(shape[39][0]))
data.append(bucket)
#right eyes 2
bucket = abs(int(shape[37][1]) - int(shape[41][1]))
data.append(bucket)
#right eyes 3
bucket = abs(int(shape[38][1]) - int(shape[40][1]))
data.append(bucket)
#left eyes 1
bucket = abs(int(shape[42][0]) - int(shape[45][0]))
data.append(bucket)
#left eyes 2
bucket = abs(int(shape[43][1]) - int(shape[47][1]))
data.append(bucket)
#left eyes 3
bucket = abs(int(shape[44][1]) - int(shape[46][1]))
data.append(bucket)
#right brow
bucket = abs(int(shape[17][1]) - int(shape[21][1]))
data.append(bucket)
#left brow
bucket = abs(int(shape[22][1]) - int(shape[26][1]))
data.append(bucket)
#nose 1
bucket = abs(int(shape[27][1]) - int(shape[30][1]))
data.append(bucket)
#nose 2
a = abs(int(shape[30][1]) - int(shape[31][1]))
b = abs(int(shape[30][0]) - int(shape[31][0]))
bucket = math.sqrt(a**2 + b**2)
data.append(bucket)
#nose 3
a = abs(int(shape[30][1]) - int(shape[35][1]))
b = abs(int(shape[30][0]) - int(shape[35][0]))
bucket = math.sqrt(a**2 + b**2)
data.append(bucket)
#mouth hor up
bucket = abs(int(shape[50][0]) - int(shape[52][0]))
data.append(bucket)
#mouth hor mid
bucket = abs(int(shape[48][0]) - int(shape[54][0]))
data.append(bucket)
#mouth hor down
bucket = abs(int(shape[58][0]) - int(shape[56][0]))
data.append(bucket)
#mouth right up
a = abs(int(shape[50][1]) - int(shape[48][1]))
b = abs(int(shape[50][0]) - int(shape[48][0]))
bucket = math.sqrt(a**2 + b**2)
data.append(bucket)
#mouth right down
a = abs(int(shape[58][1]) - int(shape[48][1]))
b = abs(int(shape[58][0]) - int(shape[48][0]))
bucket = math.sqrt(a**2 + b**2)
data.append(bucket)
#mouth left up
a = abs(int(shape[52][1]) - int(shape[54][1]))
b = abs(int(shape[52][0]) - int(shape[54][0]))
bucket = math.sqrt(a**2 + b**2)
data.append(bucket)
#mouth left down
a = abs(int(shape[56][1]) - int(shape[54][1]))
b = abs(int(shape[52][0]) - int(shape[54][0]))
bucket = math.sqrt(a**2 + b**2)
data.append(bucket)

#os.remove("testingdata.csv")
myFile = open(csvFile, "a")
with myFile:
        writer = csv.writer(myFile)
        writer.writerows([data])

#cv2.imshow('', cropped)
print("done")
cv2.waitKey(0)
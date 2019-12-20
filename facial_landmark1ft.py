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

def addData(csvFile, data):
        f = open(csvFile,'a').write([data])

image = cv2.imread('datafoto_uji/s01_latih/01.jpg')
#image = imutils.resize(image, width=500)
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

csvFile = "testingdataft.csv"
data = []
i = 0

for (i, rect) in enumerate(rects):

    shape = predictor(cropped, rect)
    shape = face_utils.shape_to_np(shape)

    #(x, y, w, h) = face_utils.rect_to_bb(rect)
    #cv2. rectangle(cropped, (x, y), (x + w, y + h), (0, 0, 0), 2)

    #cv2.putText(gray, "Face#{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)'''

    for (x,y) in shape:
        cv2.circle(cropped, (x,y), 2, (0,0,255), -1)

#area 1
#x
a = abs(int(shape[36][0]) - int(shape[39][0]))
x = abs(int(shape[39][1]) - int(shape[32][1]))
y = abs(int(shape[39][0]) - int(shape[32][0]))
b = math.sqrt(x**2 + y**2)
x = abs(int(shape[36][1]) - int(shape[32][1]))
y = abs(int(shape[36][0]) - int(shape[32][0]))
c = math.sqrt(x**2 + y**2)
z = ((c**2) + (a**2) - (b**2)) / (2*c)
#tinggi
t = math.sqrt(a**2 - z**2)
#sudut 1
bucket = np.arcsin(t/a)
data.append(bucket)
#sudut 2
bucket = np.arcsin(z/a) + np.arcsin((c-z)/b)
data.append(bucket)
#sudut 3
bucket = np.arcsin(t/b)
data.append(bucket)
#area 2
#x
x = abs(int(shape[42][1]) - int(shape[32][1]))
y = abs(int(shape[42][0]) - int(shape[32][0]))
a = math.sqrt(x**2 + y**2)
x = abs(int(shape[39][1]) - int(shape[32][1]))
y = abs(int(shape[39][0]) - int(shape[32][0]))
b = math.sqrt(x**2 + y**2)
c = abs(int(shape[39][0]) - int(shape[42][0]))
z = ((c**2) + (a**2) - (b**2)) / (2*c)
#tinggi
t = math.sqrt(a**2 - z**2)
#sudut 1
bucket = np.arcsin(t/b)
data.append(bucket)
#sudut 2
bucket = np.arcsin(t/a)
data.append(bucket)
#sudut 3
bucket = np.arcsin(z/a) + np.arcsin((c-z)/b)
data.append(bucket)
#area 3
#x
a = abs(int(shape[42][0]) - int(shape[45][0]))
x = abs(int(shape[42][1]) - int(shape[32][1]))
y = abs(int(shape[42][0]) - int(shape[32][0]))
b = math.sqrt(x**2 + y**2)
x = abs(int(shape[45][1]) - int(shape[32][1]))
y = abs(int(shape[45][0]) - int(shape[32][0]))
c = math.sqrt(x**2 + y**2)
z = ((c**2) + (a**2) - (b**2)) / (2*c)
#tinggi
t = math.sqrt(a**2 - z**2)
#sudut 1
bucket = np.arcsin(t/a)
data.append(bucket)
#sudut 2
bucket = np.arcsin(z/a) + np.arcsin((c-z)/b)
data.append(bucket)
#sudut 3
bucket = np.arcsin(t/b)
data.append(bucket)
#area 4
#x
x = abs(int(shape[32][1]) - int(shape[52][1]))
y = abs(int(shape[32][0]) - int(shape[52][0]))
a = math.sqrt(x**2 + y**2)
x = abs(int(shape[45][1]) - int(shape[32][1]))
y = abs(int(shape[45][0]) - int(shape[32][0]))
b = math.sqrt(x**2 + y**2)
x = abs(int(shape[45][1]) - int(shape[54][1]))
y = abs(int(shape[45][0]) - int(shape[54][0]))
c = math.sqrt(x**2 + y**2)
z = ((c**2) + (a**2) - (b**2)) / (2*c)
#tinggi
t = math.sqrt(a**2 - z**2)
#sudut 1
bucket = np.arcsin(t/b)
data.append(bucket)
#sudut 2
bucket = np.arcsin(z/a) + np.arcsin((c-z)/b)
data.append(bucket)
#sudut 3
bucket = np.arcsin(t/a)
data.append(bucket)
#area 5
#x
x = abs(int(shape[32][1]) - int(shape[48][1]))
y = abs(int(shape[32][0]) - int(shape[48][0]))
a = math.sqrt(x**2 + y**2)
x = abs(int(shape[36][1]) - int(shape[32][1]))
y = abs(int(shape[36][0]) - int(shape[32][0]))
b = math.sqrt(x**2 + y**2)
x = abs(int(shape[36][1]) - int(shape[48][1]))
y = abs(int(shape[36][0]) - int(shape[48][0]))
c = math.sqrt(x**2 + y**2)
z = ((c**2) + (a**2) - (b**2)) / (2*c)
#tinggi
t = math.sqrt(a**2 - z**2)
#sudut 1
bucket = np.arcsin(t/b)
data.append(bucket)
#sudut 2
bucket = np.arcsin(z/a) + np.arcsin((c-z)/b)
data.append(bucket)
#sudut 3
bucket = np.arcsin(t/a)
data.append(bucket)
#area 6
#x
x = abs(int(shape[32][1]) - int(shape[48][1]))
y = abs(int(shape[32][0]) - int(shape[48][0]))
a = math.sqrt(x**2 + y**2)
x = abs(int(shape[54][1]) - int(shape[32][1]))
y = abs(int(shape[54][0]) - int(shape[32][0]))
b = math.sqrt(x**2 + y**2)
c = abs(int(shape[54][0]) - int(shape[48][0]))
z = ((c**2) + (a**2) - (b**2)) / (2*c)
#tinggi
t = math.sqrt(a**2 - z**2) 
#sudut 1
bucket = np.arcsin(z/a) + np.arcsin((c-z)/b)
data.append(bucket)
#sudut 2
bucket = np.arcsin(t/b)
data.append(bucket)
#sudut 3
bucket = np.arcsin(t/a)
data.append(bucket)

#cv2.imwrite(file, image)
#os.remove("testingdataft.csv")
myFile = open(csvFile, "a")
with myFile:
        writer = csv.writer(myFile)
        writer.writerows([data])

#cv2.imshow('', cropped)
print("done")
cv2.waitKey(0)

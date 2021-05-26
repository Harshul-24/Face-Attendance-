import cv2 as cv
import numpy as np
import face_recognition

imgElon = face_recognition.load_image_file('/Users/User/Documents/dev/openCV/face_attendance/imagesBasic/Elonmusk.jpg')
imgElon = cv.cvtColor(imgElon,cv.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('/Users/User/Documents/dev/openCV/face_attendance/imagesBasic/Elontest.jpg')
imgTest = cv.cvtColor(imgTest,cv.COLOR_BGR2RGB)

#Encoding and recognition

#for training image - imgElon
faceLoc = face_recognition.face_locations(imgElon)[0]   #tuple of points specfiying the bounding box
encodeElon = face_recognition.face_encodings(imgElon)[0]  #set of 128 encodings unique to each face
cv.rectangle(imgElon, (faceLoc[3],faceLoc[0]), (faceLoc[1],faceLoc[2]),(255,0,0),2)

#for test image - imgTest
faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv.rectangle(imgTest, (faceLocTest[3],faceLocTest[0]), (faceLocTest[1],faceLocTest[2]),(255,0,0),2)

results  = face_recognition.compare_faces([encodeElon],encodeTest)
faceDist = face_recognition.face_distance([encodeElon],encodeTest)
print(results,faceDist)

cv.putText(imgTest, f'{results} {round(faceDist[0],2)}',(50,50),cv.FONT_HERSHEY_COMPLEX,1,(69,60,255),2)

cv.imshow('Musk',imgElon)
cv.imshow('test',imgTest)
cv.waitKey(0)
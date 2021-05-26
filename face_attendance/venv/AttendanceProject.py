import cv2 as cv
import numpy as np
import face_recognition
import os
from datetime import datetime

path = '/Users/User/Documents/dev/openCV/face_attendance/imagesAttendance'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

for cl in myList:
    curImg=cv.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

print (classNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        encodeImg = face_recognition.face_encodings(img)[0]
        encodeList.append(encodeImg)

    return encodeList


#Name time based attendance
def markAttendance(name):
    with open('/Users/User/Documents/dev/openCV/face_attendance/Attendance.csv','r+') as f:
        DataList = f.readlines()
        nameList = []  #list for storing names
        for line in DataList:
            entry = line.split(',')
            nameList.append(entry[0])   #data read in as Name Date format, and we require only name
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S:')
            f.writelines(f'\n{name},{dtString}')









encodeListKnown = findEncodings(images)
print('Encoding Complete')

#Matching encoding through webcam
cap = cv.VideoCapture(0)

while True:
    success, image = cap.read()
    imgSmall = cv.resize(image,(0,0),None, 0.25,0.25) #rescaling down the image for faster processing
    imgSmall = cv.cvtColor(imgSmall, cv.COLOR_BGR2RGB)
    #encodeImg = face_recognition.face_encodings(imgSmall)[0]

    #finding location of possible multiple faces and sending them to encoding function
    facesCurFrame = face_recognition.face_locations(imgSmall)
    encodesCurFrame = face_recognition.face_encodings(imgSmall, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace) #returns a list
        print(faceDis)

        matchIndex = np.argmin(faceDis)  #min distance is the best match, extracting that

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv.rectangle(image,(x1,y1),(x2,y2),(0,0,255),2)
            cv.rectangle(image, (x1,y2-35),(x2,y2),(0,255,0),cv.FILLED)
            cv.putText(image,name,(x1+6,y2-6), cv.FONT_HERSHEY_COMPLEX, 1, (255,255,255),2)
            markAttendance(name)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    cv.imshow('Webcam', image)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break










#faceLoc = face_recognition.face_locations(imgElon)[0]   #tuple of points specfiying the bounding box
#encodeElon = face_recognition.face_encodings(imgElon)[0]  #set of 128 encodings unique to each face
#cv.rectangle(imgElon, (faceLoc[3],faceLoc[0]), (faceLoc[1],faceLoc[2]),(255,0,0),2)

#for test image - imgTest
#faceLocTest = face_recognition.face_locations(imgTest)[0]
#encodeTest = face_recognition.face_encodings(imgTest)[0]
#cv.rectangle(imgTest, (faceLocTest[3],faceLocTest[0]), (faceLocTest[1],faceLocTest[2]),(255,0,0),2)

#results  = face_recognition.compare_faces([encodeElon],encodeTest)
#faceDist = face_recognition.face_distance([encodeElon],encodeTest)
cv.waitKey(1)


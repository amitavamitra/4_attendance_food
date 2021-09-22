import cv2
import numpy as np
import face_recognition
import os
import datetime

path = 'Images'
images = []
classNames = []
myList = os.listdir(path)
# print(myList)


for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def saveEncoding(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
        with open('encoded.csv','w+') as f:
            myCodeList = []
            for line in encodeList:
                code = str(line)
                myCodeList.append(code)
                
                f.write(f'\n,{code}')

# saveEncoding(images)

def findEncodings(images):
    encodeList = []
    for img in images:
        img =  cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('attendance.csv' ,'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

# markAttendance('Elon')


def readEncodings():
    with open('encoded.csv' ,'r+') as f:
        encodeList = f.readlines()
        codeList = []
        for line in encodeList:
            code = line.split(',')
            codeList.append(code[0])
        return codeList

encodeListKnown = findEncodings(images)
# encodeListKnown = readEncodings()
# print(len(encodeListKnown))

print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success,img = cap.read()
    imgS = cv2.resize(img, (0,0),None, 0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS , facesCurFrame)

    for encodeFace , faceLoc in zip(encodeCurFrame , facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown , encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDis)

        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            cv2.rectangle(img, (x1,y1),(x2,y2),(0,255,0),2)
            markAttendance(name)
    cv2.imshow('WebcamImage',img)
    cv2.waitKey(0)
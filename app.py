from flask import Flask,render_template,Response,session, redirect , json
# from flask_pymongo import PyMongo
# from dotenv import load_dotenv
import os
# load_dotenv()
import cv2
import numpy as np
import face_recognition
# import pandas as pd
import datetime

app=Flask(__name__)



cap = cv2.VideoCapture(0)
camera = cv2.VideoCapture(2)

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
            dtDate = now.date()
            dtString = now.strftime('%H:%M:%S')
            food ='Pizza'
            f.writelines(f'\n{name},{dtDate},{dtString},{food}')

def generate_frames():
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

    global img_counter
    global cntr

    encodeListKnown = findEncodings(images)
    # write the encoding of the faces in a  file.
    # performance improvement and also helps in Data Quality..
    # Keep all encodings in a file and read during comparison.
    # with open('encodedfaces.json', 'r') as f:
    #     encodeListKnown = f.read()
    #     encodeListKnown =   np.fromstring(encodeListKnown, dtype=float, sep='array(')
    
    while True:
            
        ## read the camera frame
        
        success,img = cap.read()
        imgS = cv2.resize(img, (0,0),None, 0.25,0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodeCurFrame = face_recognition.face_encodings(imgS , facesCurFrame)
# Compare the list with encodeface
        for encodeFace , faceLoc in zip(encodeCurFrame , facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown , encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            print(faceDis)

            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                print(name)
                y1,x2,y2,x1 = faceLoc
                y1,x2,y2,x1  = y1*4,x2*4,y2*4,x1*4 
                cv2.rectangle(img, (x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(img, (x1,y2-35),(x2,y2),(255,255,0),cv2.FILLED)
                # cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
         
                markAttendance(name)

        if not success:
            break
        else:
            ret,buffer=cv2.imencode('.jpg',img)
            
            # retu , img = cap.read()
            img=buffer.tobytes()
            
        # frame sent to the browser  
        yield(b'--img\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')

# https://stackoverflow.com/questions/52644035/how-to-show-a-pandas-dataframe-into-a-existing-flask-html-table

def gen_frames():
    
    while True:
            
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
            
            retu,buffer=cv2.imencode('.jpg',frame)
            # retu , img = cap.read()
            
            frame=buffer.tobytes()
        # frame sent to the browser  
        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
# https://stackoverflow.com/questions/52644035/how-to-show-a-pandas-dataframe-into-a-existing-flask-html-table





@app.route('/')
def index():
    
    generate_frames()
    return render_template('index.html')  

@app.route('/')
def ind():
    
    gen_frames()
    return render_template('index.html')  


@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=img')
# https://stackoverflow.com/questions/30136257/how-to-get-image-from-video-using-opencv-python


@app.route('/video_food')
def video_food():
    return Response(gen_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')
# https://stackoverflow.com/questions/30136257/how-to-get-image-from-video-using-opencv-python


if __name__=="__main__":
    app.run(debug=True)


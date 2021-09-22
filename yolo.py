import cv2
import numpy as np
import datetime
cap = cv2.VideoCapture(0)
whT=320
confThreshold = 0.7
nmsThreshold = 0.3
classesFile = 'classname.txt'
classNames = []
object = []
modelConfiguration = 'yolo320.cfg'
modelWeights = 'yolov3.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObjects(outputs , img):
    hT,wT,cT = img.shape
    bbox =[]
    classIds = []
    confs = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w,h = int(det[2]*wT) , int(det[3]*hT)
                x,y = int(det[0]*wT -w/2) , int(det[1]*hT -h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))

    print(len(bbox))
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold,nmsThreshold)
    # print(indices)
    for i in indices:
        i = i[0]
        box = bbox[i]
        x,y, w, h = box[0],box[1],box[2],box[3]
        cv2.rectangle(img, (x,y),(x+w,y+h),(255,0,255),2)
        cv2.putText(
        img,
        f'{classNames[classIds[i]].upper()}{int(confs[i]*100)}%',
        (x,y-10),
        cv2.FONT_HERSHEY_SIMPLEX,
         0.6,
         (255,0,255)
         ,2
         )
        object.append(classNames[classIds[i]])
        with open('abc.txt', 'w') as f:
            now = datetime.datetime.now()
            dtDate = now.date()
            dtString = now.strftime('%H:%M:%S')
            food =str(object)
            f.writelines(f'\n{food},{dtDate},{dtString}')
            # f.write(str(object), datetime.now()) )

with open(classesFile , 'r') as f:
    classNames = f.read().rstrip('\n').split('\n')




while True:
    success,img = cap.read()
    blob = cv2.dnn.blobFromImage(img , 1/255, (whT , whT),[0,0,0],1,crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    # print(layerNames)
    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    # print(outputNames)

    outputs = net.forward(outputNames)
    # print(outputs[0].shape)
    # print(outputs[1].shape)
    # print(outputs[2].shape)
    # print(outputs[0][0])
    
    findObjects(outputs,img)

    cv2.imshow("Image", img)
    cv2.waitKey(10)
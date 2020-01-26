import numpy as np
import cv2
import os
import time
cam = cv2.VideoCapture(0)
faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
rec=cv2.face.LBPHFaceRecognizer_create()
rec.read("recognizer/training_data.yml")
i = 0
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output'+str(i)+'.avi', fourcc, 10.0, (640,480))
while True:
    i = 0
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        if(id==1):
            text="ILHAM"
        else:
            text="UNKNOWN"
        cv2.putText(img, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0),10)
        if os.path.exists("output"+str(i)+".avi"):
            i += 1
        out.write(img)
    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
out.release()
cv2.destroyAllWindows()

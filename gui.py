import cv2
import os
from keras.models import load_model
from pygame import mixer
import numpy as np

mixer.init()
alarm = mixer.Sound("alarm.wav")

face = cv2.CascadeClassifier("drowssines\haar cascade files\haarcascade_frontalface_alt.xml")
left_eye = cv2.CascadeClassifier("drowssines\haar cascade files\haarcascade_lefteye_2splits.xml")
right_eye = cv2.CascadeClassifier("drowssines\haar cascade files\haarcascade_righteye_2splits.xml")

label = ["Close", "Open"]
model = load_model("drowssines\models\cnnCat22.h5")
path = os.getcwd()
capt = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count= 0
score = 0
thicc = 2
rpred = [99]
lpred = [99]

while True:
    ret, frame = capt.read()
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor = 1.1, minSize = (25,25))
    l_eye = left_eye.detectMultiScale(gray)
    r_eye= right_eye.detectMultiScale(gray)
    cv2.rectangle(frame,(0,height-50), (200, height), (0,0,0), thickness=cv2.FILLED)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(100,100,100), 1)

    for (x,y,w,h) in l_eye:
        leye = frame[y:y+h, x:x+w]
        leye = cv2.cvtColor(leye,cv2.COLOR_BGR2GRAY)
        leye = cv2.resize(leye, (24,24))
        leye = leye/255
        leye = leye.reshape(24,24,-1)
        leye = np.expand_dims(leye,axis= 0)
        lpred = np.argmax(model.predict(leye),axis = -1)
        if lpred[0] == 1:
            label = "Open"
        if lpred[0] == 0:
            label = "Closed"
        break

    for (x,y,w,h) in r_eye:
        reye = frame[y:y+h, x:x+w]
        reye = cv2.cvtColor(reye,cv2.COLOR_BGR2GRAY)
        reye = cv2.resize(reye, (24,24))
        reye = reye/255
        reye = reye.reshape(24,24,-1)
        reye = np.expand_dims(reye,axis= 0)
        rpred = np.argmax(model.predict(reye),axis = -1)
        if rpred[0] == 1:
            label = "Open"
        if rpred[0] == 0:
            label = "Closed"
        break

    if rpred[0] == 0 and lpred[0] == 0:
        score += 1
        cv2.putText(frame,"Closed", (10,height-20), font, 1, (255,255,255), 1, cv2.LINE_AA)
    else:
        score-=1
        cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if score < 0:
        score = 0

    cv2.putText(frame, "Score: " + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if score >= 15:
        cv2.imwrite(os.path.join(path,"Image_driving.jpg"),frame)
        try:
            alarm.play()
        except:
            pass
        if thicc < 16:
            thicc += 2
        else:
            thicc -= 2
            if thicc<2:
                thicc = 2
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

    cv2.imshow("closed",frame)
    if cv2.waitKey(1) & 0xff == ord("q"):
        break

capt.release()
cv2.destroyAllWindows()



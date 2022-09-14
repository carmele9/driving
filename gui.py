import cv2
import os
from keras.models import load_model
from pygame import mixer

mixer.init()
alarm = mixer.Sound("alarm.wav")

face = cv2.CascadeClassifier("drowssines\haar cascade files\haarcascade_frontalface_alt.xml")
left_eye = cv2.CascadeClassifier("drowssines\haar cascade files\haarcascade_lefteye_2splits.xml")
right_eye = cv2.CascadeClassifier("drowssines\haar cascade files\haarcascade_righteye_2splits.xml")



import os, re, glob
import cv2
import numpy as np
import shutil
from numpy import argmax
from keras.models import load_model

categories = ["0", "1", "2", "3", "4", "5", "6", "7"]


def Dataization(img_path):
    image_w = 128
    image_h = 100
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=image_w/img.shape[1], fy=image_h/img.shape[0])
    return (img/256)

src = []
name = []
test = []
image_dir = "c:\\users\\gurrl\\test\\1\\"
for file in os.listdir(image_dir):
    src.append(image_dir + file)
    name.append(file)
    test.append(Dataization(image_dir + file))



test = np.array(test)
model = load_model('c:\\users\\gurrl\\model_dress_1.h5')
predict = model.predict_classes(test)

for i in range(len(test)):
    print(name[i] + " : , Predict : "+ str(categories[predict[i]]))

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from IPython.display import Image
import pandas as pd
import numpy as np
import pydotplus
import os
import requests
import json
import time


tennis_data = pd.read_csv('C:/users/gurrl/test.csv')
print(tennis_data)

tennis_data.dir = tennis_data.dir.replace('0',0)
tennis_data.dir = tennis_data.dir.replace('1', 1)
tennis_data.dir = tennis_data.dir.replace('2',2)
tennis_data.dir = tennis_data.dir.replace('3',3)
tennis_data.dir = tennis_data.dir.replace('4', 4)
tennis_data.dir = tennis_data.dir.replace('5',5)
tennis_data.dir = tennis_data.dir.replace('6',6)
tennis_data.dir = tennis_data.dir.replace('7', 7)


tennis_data.jender = tennis_data.jender.replace('man',8)
tennis_data.jender = tennis_data.jender.replace('girl',9)

tennis_data.atom = tennis_data.atom.replace('yes',10)
tennis_data.atom = tennis_data.atom.replace('no',11)

tennis_data.summer = tennis_data.summer.replace('yes',12)
tennis_data.summer = tennis_data.summer.replace('no',13)

tennis_data.fall = tennis_data.fall.replace('yes',14)
tennis_data.fall = tennis_data.fall.replace('no',15)

tennis_data.winter = tennis_data.winter.replace('yes',16)
tennis_data.winter = tennis_data.winter.replace('no',17)


print(tennis_data)

X = np.array(pd.DataFrame(tennis_data, columns=['jender','atom','summer', 'fall','winter']))
y = np.array(pd.DataFrame(tennis_data, columns=['dir']))
X_train, X_test, y_train, y_test = train_test_split(X,y)

# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)

dt_clf = DecisionTreeClassifier()
dt_clf = dt_clf.fit(X_train, y_train)
dt_prediction = dt_clf.predict(X_test)

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
feature_names = tennis_data.columns.tolist()
feature_names = feature_names[0:5]

target_name = np.array(['dir 0', 'dir 1','dir 2', 'dir 3','dir 4','dir 5', 'dir 6', 'dir 7'])

dt_dot_data = tree.export_graphviz(dt_clf, out_file = None,
                                  feature_names = feature_names,
                                  class_names = target_name,
                                  filled = True, rounded = True,
                                  special_characters = True)

dt_graph = pydotplus.graph_from_dot_data(dt_dot_data)
Image(dt_graph.create_png())
# 데이터 잔처리 남8 여9/10 13 15 17 봄  ,11 12 15 17 여름, 11 13 14 17 가을 ,11 13 15 16 겨울

while True:

    p1 = requests.get('http://192.168.18.1:3000/clo/cls').text
    jsonObject1 = json.loads(p1)
    jsonArray1= jsonObject1.get("user")
    print(jsonArray1)

    for per1 in jsonArray1:
        p1 =(per1.get("jender"))
    for per1 in jsonArray1:
        p2 =(per1.get("season"))
    for per1 in jsonArray1:
        st =(per1.get("on_off"))
    if st == '1':
        print(p1,","+p2)
        reco = p1+","+p2
        reco
        st
        p3 = str(p2)[0:2]
        p4 = str(p2)[3:5]
        p5 = str(p2)[6:8]
        p6 = str(p2)[9:11]
        print(p1)

        a = dt_clf.predict([[p1,p3,p4,p5,p6]])
        print(a)
#머신러닝 결과 값

        p = "c:\\users\\gurrl\\project\\project\\uploads\\" + str(a)[1:2]+"\\"
# 경로 변수화
        import os, re, glob
        import cv2
        import numpy as np
        import shutil
        from numpy import argmax
        from keras.models import load_model

# 딥러닝 모델 활용을 통한 이미지 분류
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
        image_dir = p
        for file in os.listdir(image_dir):
            src.append(image_dir + file)
            name.append(file)
            test.append(Dataization(image_dir + file))



        test = np.array(test)
        model = load_model('c:\\users\\gurrl\\model_dress_1.h5')
        predict = model.predict_classes(test)


        for i in range(len(test)):
            print(name[i] + src[i]+ " : , Predict : "+ str(categories[predict[i]]))


        import random
        for i in range(8) :
            n = random.randrange(0,30)
            print(n)
            # img_name = p + '/' + name[n]
            # img = cv2.imread(img_name, cv2.IMREAD_COLOR)
            # cv2.imshow('image', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            data1 = {'src': str(src[i])[31:]}
            res1 = requests.post('http://192.168.137.1:3000/clo/cls1', data=data1)

        data = {'on_off': '0'}
        res = requests.post('http://192.168.137.1:3000/clo/cls', data=data)
        # os.system('explorer http://192.168.137.1:3000/recom/re')
    else:
        time.sleep(3)
        continue;
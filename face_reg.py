import cv2 as cv
import numpy as np
import os
import serial
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import pickle
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity

port=serial.Serial('COM3',9600)
facenet = FaceNet()
faces_embeddings = np.load("./faces_embeddings_done_4classes.npz")
Y = faces_embeddings['arr_1']
faces_embeddings=faces_embeddings['arr_0']
haarcascade = cv.CascadeClassifier("./haarcascade_frontalface_default.xml")

cap = cv.VideoCapture(0)

similarity_threshold = 0.7

while cap.isOpened():
    _, frame = cap.read()
    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)
    for x,y,w,h in faces:
        img = rgb_img[y:y+h, x:x+w]
        img = cv.resize(img, (160,160)) # 1x160x160x3
        img = np.expand_dims(img,axis=0)
        new_face_embedding = facenet.embeddings(img)
        similarity_scores = cosine_similarity(new_face_embedding, faces_embeddings)
        max_index = np.argmax(similarity_scores)
    

        if similarity_scores[0][max_index] > similarity_threshold:

            identified_person = Y[max_index]
            cv.rectangle(frame, (x,y), (x+w,y+h), (255,0,255), 10)
            cv.putText(frame, str(identified_person), (x,y-10), cv.FONT_HERSHEY_SIMPLEX,
                   1, (0,0,255), 3, cv.LINE_AA,)
            print(f'Identified person: {identified_person}')
            port.write(str.encode('1'))
    
        else:
            print('No identified person.')
            port.write(str.encode('0'))

    cv.imshow("Face Recognition:", frame)
    if cv.waitKey(1) & ord('q') ==27:
        break

cap.release()
cv.destroyAllWindows
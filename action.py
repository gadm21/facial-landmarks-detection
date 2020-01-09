from model import *
import cv2
import numpy as np
from config import *
from helper import preprocess_frame, preprocess_face


#load model
model= LoadModel("facelandmarks_model")

#face cascade to detect cases
face_cascade= cv2.CascadeClassifier('cascades/haarcascade_frontalface.xml')

#load the video
camera= cv2.VideoCapture(0)


#iterate over video frames
while True:
    #grab current frame
    (found, frame)= camera.read()
    frame_size= frame.shape[:2]

    if not found: break

    

    
    
    flipped_frame, hsv, gray= preprocess_frame(frame)

    #detect faces, returns their coords.
    faces= face_cascade.detectMultiScale(gray, 1.02, 4)
    
    for face in faces:
        (x,y,w,h)= face

        # Grab the face
        gray_face = gray[y:y+h, x:x+w]
        color_face = frame[y:y+h, x:x+w]    

        #save face shape
        face_shape= gray_face.shape

        #normalization and resizing
        preprocessed_face= preprocess_face(gray_face)


        #predict face landmarks
        landmarks= model.predict(preprocessed_face)

        #convert landmarks from [-1, 1] to [0, 96]
        landmarks= landmarks * 48 + 48

        points = []
        for i, co in enumerate(landmarks[0][0::2]):
            points.append((int(co), int(landmarks[0][1::2][i])))


        #resize, draw landmarks, and resize back
        color_face= cv2.resize(color_face, (96, 96), interpolation= cv2.INTER_AREA)
        for point in points:
            cv2.circle(color_face, point, 1, (255, 100, 0), 2)
        frame[y:y+h, x:x+w]= cv2.resize(color_face, face_shape, interpolation= cv2.INTER_AREA)

        cv2.rectangle(frame, (y, x), (y+h, x+w), (255, 255, 0), 2)
        
        # print("first point:", points[0])
        # print("color_face:", color_face.shape)
        # print("face_shape:", face_shape)
        # print("frame_size:", frame_size)
        # print("true frame_size:", frame.shape)

        # print(".....................................................")
        # print(".....................................................")
        # print(".....................................................")

        #show
    
    cv2.imshow("image with landmarks", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

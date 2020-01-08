from model import *
import cv2
import numpy as np
from config import *
from helper import preprocess_frame, get_contour_center, preprocess_face

#load model
model= LoadModel("facelandmarks_model")

#face cascade to detect cases
face_cascade= cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

#load the video
camera= cv2.VideoCapture(0)

#iterate over video frames
while True:
    #grab current frame
    (found, frame)= camera.read()
    if not found: break

    frame_size= frame.shape[:1]
    flipped_frame, hsv, gray= preprocess_frame(frame)

    #draw a button on the frame
    cv2.rectangle(frame, (500, 10), (620, 65), (235, 50,50), -1)
    cv2.putText(frame, "gad is a hero", (512, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)

    #detect faces, returns their coords.
    faces= face_cascade.detectMultiScale(gray, 1.25, 6)

    #get center of the blue contour (bottle cap in this case)
    center= get_contour_center(hsv)

    if center[1] <= 65:
        if 500 <= center[0] <= 620:
            filter_index= (filter_index+1) % 6
            continue
    
    for face in faces
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

        '''
        since each of the andmarks value has from [-1, 1] representing the 
        x or y of the landmark coordinates in the image, we will
        normalize it by converting it from [-1, 1] to [0, 96]
        '''
        landmarks= landmarks * 48 + 48



        points = []
        for i, co in enumerate(landmarks[0][0::2]):
            points.append((co, landmarks[0][1::2][i]))

        #resize, draw landmarks, and resize back
        color_face= cv2.resize(color_face, frame_size, interpolation= INTER_AREA)
        for point in points:
            cv2.circle(color_face, point, 1, (255, 255, 100), 1)
        frame[y:y+h, x:x+w]= cv2.resize(color_face, face_shape, interpolation= INTER_AREA)


        #show
        cv2.imshow("image with landmarks", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

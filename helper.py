import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from keras.models import load_model
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle

from config import *

def load_facelandmarks_data(file_name, test= False):
    '''
    file should be a .csv file

    loads data from testfile if test= True. Otherwise, loads data from train_file
    Although file_name is provided, the function needs to know whether this is
    the training or testing data because it will parse them differently
    '''

    file= read_csv(os.path.expanduser(file_name)) #loads dataframes

    '''
    the image column has pixel values separated by space,
    here we convert these values to numpy arrays
    '''
    file['image']= file['image'].apply(lambda im: np.fromstring(im, sep= ' '))

    file= file.dropna() #drop all empty rows

    x= np.vstack(file['image'].values) /255 #normalize, scale pixel values to [0,1]
    x= x.astype(np.float32)
    x= x.reshape(-1, 96, 96, 1) #return each image as 96 x 96 x 1

    if not test: #only train_file has target columns
        y= file[file.columns[:-1]].values
        y= (y-48)/48 # Normalize, scale target coordinates to [-1,1]
        x,y= shuffle(x, y, random_state= 42) #shuffle train data 
        y= y.astype(np.float32)
    else:
        y= None
    
    return x, y


def preprocess_frame(frame):
    flipped_frame= cv2.flip(frame, 1)
    hsv= cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return flipped_frame, hsv, gray

def extract_blue_mask(hsv_frame):
    blue_mask= cv2.inRange(hsv_frame, blue_lower, blue_upper)
    blue_mask= cv2.erode(blue_mask, kernel, iterations= 2)
    blue_mask= cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    blue_mask= cv2.dilate(blue_mask, kernel, iterations= 1)

    return blue_mask

def get_contour_center(hsv_frame):

    #determine which pixels fall withing the blue boundaries
    blue_mask= extract_blue_mask(hsv_frame)

    #find contours in the image (bottle cap in this case)
    (_, contours, _)= cv2.findContours(blue_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center= None

    #if any contours was found
    if len(contours) > 0:

    #find the largest contour
    largest_contour= sorted(contours, key= cv2.contourArea, reverse= True)[0]

    #get the radius of the circle around the contour
    ((x,y), radius)= cv2.minEnclosingCircle(largest_contour)

    #draw the circle around the contour
    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)

    # Get the moments to calculate the center of the contour (in this case Circle)
    M = cv2.moments(largest_contour)

    center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

    return center


def preprocess_face(gray_face):
    normalized_face= gray_face/255
    resized_face= cv2.resize(normalized_face, (96, 96), interpolation= cv2.INTER_AREA)
    reshaped_face= resized_face.reshape(1, 96, 96, 1)

    return reshaped_face

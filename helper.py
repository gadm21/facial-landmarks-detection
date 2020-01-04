import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from keras.models import load_model
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle

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


from keras.models import Sequential
from keras.models import load_model
from keras.layers import Convolution2D, MaxPooling2D, Dropout
from keras.layers import Flatten, Dense
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam

'''
this project uses this dataset on kaggle :: https://www.kaggle.com/c/facial-keypoints-detection
to train a CNN to detect facial keypoints given an image of a face.


In this file, we define the CNN model architecure which we will
train on kaggle's face-landmarks dataset.

'''


def my_CNN_model():
    model= Sequential
    model.add(Conv)


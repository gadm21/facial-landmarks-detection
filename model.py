'''
this project uses this dataset on kaggle :: https://www.kaggle.com/c/facial-keypoints-detection
to train a CNN to detect facial keypoints given an image of a face.


In this file, we define the CNN model architecure which we will
train on kaggle's face-landmarks dataset.

'''

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Convolution2D, MaxPooling2D, Dropout
from keras.layers import Flatten, Dense
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam

# def warn(*args, **kwargs):
#     pass
# import warnings
# warnings.warn = warn




def FaceLandmarks_CNN_model(input_shape= 96, channels= 1):
    model= Sequential()

    model.add(Convolution2D(32, (5,5), input_shape= (input_shape, input_shape, channels), activation= "relu"))
    model.add(MaxPooling2D(pool_size= (2,2)))
    model.add(Dropout(0.1))

    model.add(Convolution2D(64, (3,3), activation= "relu"))
    model.add(MaxPooling2D(pool_size= (2,2)))
    model.add(Dropout(0.1))

    model.add(Convolution2D(128, (3,3), activation= "relu"))
    model.add(MaxPooling2D(pool_size= (2,2)))
    model.add(Dropout(0.2))   

    model.add(Convolution2D(30, (3,3), activation= "relu"))
    model.add(MaxPooling2D(pool_size= (2,2)))
    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(64, activation= "relu"))
    model.add(Dense(128, activation= "relu"))
    model.add(Dense(256, activation= "relu"))
    model.add(Dense(64, activation= "relu"))
    model.add(Dense(30))

    return model

def compile_model(model, optimizer, loss, metrics):
    model.compile(optimizer= optimizer, loss= loss, metrics= metrics, validation_split= 0.2)

def train_model(model, x_train, y_train, epochs= 100, batch_size= 200, verbose= False):
    return model.fit(x_train, y_train, epochs= epochs, batch_size= batch_size, verbose= verbose, validation_split= validation_split)

def SaveModel(model, file_name):
    model.save(file_name+".h")

def LoadModel(file_name):
    return load_model(file_name+".h")

m= FaceLandmarks_CNN_model()
print("hello")

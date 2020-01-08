from helper import load_facelandmarks_data
from model import FaceLandmarks_CNN_model, compile_model, train_model, SaveModel
import cv2

#load training set
x_train, y_train= load_facelandmarks_data("data/train.csv")

#getting the CNN_model
model= FaceLandmarks_CNN_model()

#compiling model
compile_model(model, optimizer= "adam", loss= "mean_squared_error", metrics= ["accuracy"])

#training model
'''
this function calls model.fit which returns a history object which contains
a record of training loss values and metrics values at successive epochs, 
as well as validation loss values and validation metrics values.
'''
hist= train_model(model, x_train, y_train, validation_split= 0.2)

SaveModel(model, "facelandmarks_model")


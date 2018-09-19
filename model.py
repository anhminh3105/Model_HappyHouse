#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from model_utils import *

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt

def HappyModel(input_shape):
    """
    Implementation of the HappyModel.
    
    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((3, 3))(X_input)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name='max_pool40')(X)

    X = Conv2D(64, (5, 5), strides = (1, 1), padding='same', name = 'conv1')(X)
    X = Dropout(rate=0.1)(X)
    X = BatchNormalization(axis = 3, name = 'bn1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name='max_pool1')(X)

    X = Conv2D(128, (3, 3), strides = (1, 1), name = 'conv2')(X)
    X = Dropout(rate=0.1)(X)
    X = BatchNormalization(axis = 3, name = 'bn2')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name='max_pool2')(X)

    X = Conv2D(256, (3, 3), strides = (1, 1), name = 'conv3')(X)
    X = Dropout(rate=0.1)(X)
    X = BatchNormalization(axis = 3, name = 'bn3')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name='max_pool3')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='HappyModel')    

    return model

def TestHappyImg(model, img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    plt.imshow(img)
    plt.show()
    x = image.img_to_array(img)
    x = np.expand_dims(x, 0)
    x = preprocess_input(x)
    prediction = int(np.squeeze(model.predict(x)))
    print(prediction)
    if prediction:
        result = "HAPPY."
    else:
        result = "NOT HAPPY."

    print("Model thinks the person is " + result)

def main():
    
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

    # Normalize image vectors
    X_train = X_train_orig/255.
    X_test = X_test_orig/255.

    # Reshape
    Y_train = Y_train_orig.T
    Y_test = Y_test_orig.T

    print ("number of training examples = " + str(X_train.shape[0]))
    print ("number of test examples = " + str(X_test.shape[0]))
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(Y_train.shape))
    print ("X_test shape: " + str(X_test.shape))
    print ("Y_test shape: " + str(Y_test.shape))
    print ("Classes: ", str(classes))
    
    # Initialise the model.
    happyModel = HappyModel(X_train.shape[1:])

    # Compile the model.
    happyModel.compile(loss='mean_squared_error', optimizer='adadelta', metrics=['accuracy'])

    # Start the training process.
    happyModel.fit(x=X_train, y=Y_train, epochs=25, batch_size=64)

    # Test the accuracy of the model on the Test dataset.
    preds = happyModel.evaluate(x=X_test, y=Y_test)

    # Print the test accuracy.
    print()
    print ("Loss = " + str(preds[0]))
    print ("Test Accuracy = " + str(preds[1]))
    
    # Predict a random example.
    img_path = "images/my_image.jpg"
    TestHappyImg(happyModel, img_path)
    
    # print the summary of the model structure.
    happyModel.summary()

    # plot the layers in a nice layout.
    plot_model(happyModel, to_file="HappyModel.png")
    SVG(model_to_dot(happyModel).create(prog="dot", format="svg"))

    img = image.load_img("HappyModel.png")
    plt.imshow(img)
    plt.show()
if __name__ == '__main__':
    main()
    
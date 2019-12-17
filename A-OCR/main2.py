import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Activation, Embedding, Conv1D, Conv2D, GlobalMaxPooling1D, Flatten

## Path to dataset
path = '../dataset/ahcd1'

## read train data
X_train = pd.read_csv(path + '/csvTrainImages.csv')
Y_train = pd.read_csv(path + '/csvTrainLabel.csv')
## print train data
print('Train data: ')
# print(X_train.describe())
# print(X_train.head)

X_train = np.asarray(X_train)
X_train = X_train.reshape(13439, 1024, 1)

Y_train = to_categorical(Y_train)
print(X_train.shape, Y_train.shape)


## read test data
X_test  = pd.read_csv(path + '/csvTestImages.csv')
Y_test  = pd.read_csv(path + '/csvTestLabel.csv')
## print test data
print('Test data: ')
# print(X_test.describe())
# print(X_test.head)

X_test = np.asarray(X_test)
X_test = X_test.reshape(3359, 1024, 1)

Y_test = to_categorical(Y_test)
print(X_test.shape, Y_test.shape)


## Main task of CNN is to: extract features from tha training data
## Convolution model parameters:-
##   filters: dimensionality of the output space
##   kernel_size: length of the 1D convolution window
## The softmax function squashes the outputs of each unit to be between 0 and 1,
## it also divides each output such that the total sum of the outputs is equal to 1
## The relu function remove all negative values, so that the output between (0, max)


## Create Model
model = Sequential()
## Add Model Layers
model.add(Conv1D(256, kernel_size=3, activation='relu'))
model.add(Conv1D(128, kernel_size=3, activation='relu'))
model.add(Conv1D(64, kernel_size=3, activation='relu'))
model.add(Conv1D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(29, activation='softmax'))

## Compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

## train the model
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=4)

## Print Accuracy of model
scores = model.evaluate(X_test, Y_test)
print("Accuracy: %.2f%%" % (scores[1]*100))

## summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

## summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

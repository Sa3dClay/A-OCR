import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Activation, Embedding, Conv1D, Conv2D, GlobalMaxPooling1D, Flatten

## Path to dataset
path = '../dataset/ahdd1'

## read train data
X_train = pd.read_csv(path + '/csvTrainImages.csv')
Y_train = pd.read_csv(path + '/csvTrainLabel.csv')
## print train data
print('Train data: ')
# print(X_train.describe())
# print(X_train.head)

X_train = np.asarray(X_train)
X_train = X_train.reshape(59999,784,1)

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
X_test = X_test.reshape(9999,784,1)

Y_test = to_categorical(Y_test)
print(X_test.shape, Y_test.shape)


# Create Model
model = Sequential()
# Add Model Layers
model.add(Conv1D(64, kernel_size=3, activation='relu'))
model.add(Conv1D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

# Compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#train the model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=4)



# adapted from https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils



# The data, shuffled and split between train and test sets:
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print('X_test shape:', X_test.shape)
print('y_test shape:', y_test.shape)


nb_classes = 10

# Convert class vectors to binary class matrices.
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

#Flatten each image into a flat vector
inputdim = X_train.shape[1]*X_train.shape[2]*X_train.shape[3]
X_train = X_train.reshape(X_train.shape[0], inputdim)
X_test = X_test.reshape(X_test.shape[0], inputdim)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print('y_train shape:', Y_train.shape)
print('X_test shape:', X_test.shape)
print('y_test shape:', Y_test.shape)
print (X_train.shape[1:])


# In[4]:

model = Sequential()
model.add(Dense(nb_classes, input_shape=X_train.shape[1:]))
model.add(Activation('softmax'))

model.summary()


# In[8]:

sgd = SGD(lr=0.001)
#model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    batch_size=32, nb_epoch=5,
                    verbose=1, validation_data=(X_test, Y_test))


# In[6]:

score = model.evaluate(X_test, Y_test, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])


# In[ ]:

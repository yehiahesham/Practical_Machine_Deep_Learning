

# adapted from https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Activation,Dropout ,Flatten
from keras.optimizers import SGD,Adam
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras.constraints import maxnorm
from keras.models import load_model
from keras.callbacks import ModelCheckpoint,TensorBoard
import keras


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








#Flatten each image into a flat vector, used in testing
# inputdim = X_train.shape[1]*X_train.shape[2]*X_train.shape[3]
# X_train = X_train.reshape(X_train.shape[0], inputdim)
# X_test = X_test.reshape(X_test.shape[0], inputdim)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# X_train /= 255
# X_test /= 255
# X_test -= np.mean(X_test, axis=0)



print('X_train shape:', X_train.shape)
print('y_train shape:', Y_train.shape)
print('X_test shape:', X_test.shape)
print('y_test shape:', Y_test.shape)
print (X_train.shape[1:])



# model = load_model('./weights_1500.hdf5')

# In[4]:

model = Sequential()

#input layer
model.add(Flatten(input_shape=X_train.shape[1:]))
model.add(Dense(1200, activation='relu',W_constraint=maxnorm(1)))
#model.add(Dropout(0.2))

#hidden layers
model.add(Dense(512, activation='relu',W_constraint=maxnorm(1)))
#model.add(Dropout(0.2))

model.add(Dense(128, activation='relu',W_constraint=maxnorm(1)))
model.add(Dropout(0.5))

#output layer
model.add(Dense(nb_classes, activation='softmax'))
model.summary()



batchSize = 128 #32
learning_rate=0.0001 #5.586261e-04   #0.0001
epochs=1500 #200

sgd = SGD(lr=learning_rate, momentum=0.7,nesterov=True)
#model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
adam=Adam(lr=learning_rate, beta_1=0.7, beta_2=0.999, epsilon=1e-08, decay=0.0000001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])



# history = model.fit(X_train, Y_train,
#                     batch_size=batchSize, nb_epoch=epochs,
#                     verbose=1, validation_data=(X_test, Y_test))
        #   model.fit_generator(datagen.flow(train_features, train_labels, batch_size = 128),
        #                          samples_per_epoch = train_features.shape[0], nb_epoch = 200,
        #                          validation_data = (test_features, test_labels), verbose=0)

# history = model.fit_generator(datagen.flow(X_train, Y_train,batch_size=batchSize),
#                     samples_per_epoch=len(X_train.shape[0],nb_epoch=epochs,
#                     validation_data=(X_test, Y_test),verbose=1 )

datagen2 = ImageDataGenerator(
    featurewise_center=True,
    samplewise_center=False,
    featurewise_std_normalization=True,
    samplewise_std_normalization=False)
datagen2.fit(X_test)

datagen = ImageDataGenerator(
    featurewise_center=True,
    samplewise_center=False,
    featurewise_std_normalization=True,
    samplewise_std_normalization=False,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

datagen.fit(X_train)

board = keras.callbacks.TensorBoard(log_dir='./logs/logs_1500V2', histogram_freq=0, write_graph=True, write_images=False)
checkpointer = ModelCheckpoint(filepath="./weights/weights_1500.hdf5", verbose=1, save_best_only=True)




history = model.fit_generator(datagen.flow(X_train, Y_train,batch_size=batchSize),
                    steps_per_epoch=X_train.shape[0] / batchSize ,
                    epochs=epochs,validation_data=datagen2.flow(X_test, Y_test,batch_size=128 ),nb_val_samples=X_test.shape[0],verbose=1, callbacks=[board,checkpointer] )

# In[6]:
# datagen2.flow(....,batch_size=128) ,nb_val_samples=X_test.shape[0]
score = model.evaluate(X_test, Y_test, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])


# In[ ]:

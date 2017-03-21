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
from keras import backend as K

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







X_train = X_train.astype('float32')
X_test = X_test.astype('float32')



print('X_train shape:', X_train.shape)
print('y_train shape:', Y_train.shape)
print('X_test shape:', X_test.shape)
print('y_test shape:', Y_test.shape)
print (X_train.shape[1:])




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




#Loading best weights
model.load_weights("./weights/weights_1500.hdf5")




batchSize = 128 #32
learning_rate=0.0001 #5.586261e-04   #0.0001
epochs=0 #200

# sgd = SGD(lr=learning_rate, momentum=0.7,nesterov=True)
adam=Adam(lr=learning_rate, beta_1=0.7, beta_2=0.999, epsilon=1e-08, decay=0.0000001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])


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

board = keras.callbacks.TensorBoard(log_dir='./logs/logs_1500V3', histogram_freq=0, write_graph=True, write_images=False)
checkpointer = ModelCheckpoint(filepath="./weights/weights_1500V3.hdf5", verbose=1, save_best_only=True, monitor="val_acc")



history = model.fit_generator(datagen.flow(X_train, Y_train,batch_size=batchSize),
                    steps_per_epoch=X_train.shape[0] / batchSize ,
                    epochs=epochs,validation_data=datagen2.flow(X_test, Y_test,batch_size=128 ),nb_val_samples=X_test.shape[0],verbose=1, callbacks=[board,checkpointer] )



#manual (preprocessing of data )avg and normalizeation of testing data

channel_axis=0
col_axis=0
row_axis=0

data_format = K.image_data_format()
if data_format == 'channels_first':
    channel_axis = 1
    row_axis = 2
    col_axis = 3
if data_format == 'channels_last':
    channel_axis = 3
    row_axis = 1
    col_axis = 2

mean = np.mean(X_test, axis=(0, row_axis, col_axis))
std  = np.std(X_test, axis=(0, row_axis, col_axis))
broadcast_shape = [1, 1, 1]
broadcast_shape[channel_axis - 1] = X_test.shape[channel_axis]
mean = np.reshape(mean, broadcast_shape)
std  = np.reshape(std, broadcast_shape)
X_test -= mean
X_test /= (std + K.epsilon())



predicted_classes = model.predict_classes(X_test, batch_size=1, verbose=1)
reshaped_y_test=np.reshape(y_test,(10000,))
num_correct = np.sum(predicted_classes == reshaped_y_test)

print (num_correct)
acc = float(num_correct)/10000.0
print(acc*100)
Classes= ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

X=[]
for c in range(len(Classes)):
    X.append(np.sum(predicted_classes[i]==c  and  reshaped_y_test[i]==c for i in range(reshaped_y_test.shape[0])))

correctPerClass = np.array(X)

for c in range(len(Classes)):
    print ('Got ', correctPerClass[c] , ' on class ',Classes[c], ' which is ', correctPerClass[c]/1000.0)

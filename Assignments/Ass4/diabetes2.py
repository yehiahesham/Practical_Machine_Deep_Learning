from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from keras.preprocessing import image
from keras.models import Model
from keras.optimizers import SGD, Adam, Nadam, RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Activation,Dropout ,Flatten
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras.constraints import maxnorm
from keras.models import load_model
from keras.callbacks import ModelCheckpoint,TensorBoard, ReduceLROnPlateau
from keras.applications import Xception
from keras import backend as K
import keras
import csv
import os
import pandas as pd
import numpy as np

seed = 2558
np.random.seed(seed)


def loadData(path):     #loads data , caluclate Mean & subtract it data, gets the COV. Matrix.
    D = pd.read_csv(path)
    feature_names  = np.array(list(D.columns.values))
    Y_train = np.array(list(D['Class']));
    X_train=D.ix[:,:8]
    mean = X_train.mean()
    X_train=X_train-X_train.mean();
    cov =  X_train.cov();

    print 'X_train.shape is ', X_train.shape
    print 'Y_train.shape is ',Y_train.shape
    print 'cov.shape is ',cov.shape
    print 'Feature are ',feature_names
    return  X_train, Y_train, mean, cov, feature_names

X_train, Y_train, mean, cov, feature_names = loadData('./data/pima-indians-diabetes.csv')

eigenvalues, eigenvectors = np.linalg.eig(cov);

print 'eigenvalues.shape is',eigenvalues.shape
print 'eigenvectors.shape is ',eigenvectors.shape # they are normalized already, IE each is a unit vector

print Y_train.shape




# One Hot Encoding
Y_train = keras.utils.to_categorical(Y_train, 2)


# sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
batchSize = 128 #32
learning_rate=0.0001 #5.586261e-04   #0.0001
epochs=50 #200
num_folds = 5
dims_kept= [1,2,3,4,5,6,7,8]
kf = KFold(n_splits=num_folds)
# newEigensIndices = np.argsort(-eigenvalues)[:dims]
sortedEigensIndices = np.argsort(-eigenvalues)

accuracies=[]

for dims in dims_kept:
    print '\t# Dimensions to Keep is ',dims
    newEigensIndices = sortedEigensIndices[:dims]
    Proj_X_train = np.dot(X_train.as_matrix(),eigenvectors[newEigensIndices].T);
    print '\tProj_X_train.shape is ',Proj_X_train.shape

    for train, test in kf.split(X_train.as_matrix()):
        kX_train, kX_test, kY_train, kY_test = X_train.as_matrix()[train], X_train.as_matrix()[test], Y_train[train], Y_train[test]

        # Define the classifier based on the # of Dimensions
        model = Sequential()
        model.add(Dense(8, input_shape=(kX_train.shape[1:])))
        #model.add(Dropout(0.2))

        #hidden layers
        # model.add(Dense(64, activation='relu')) #,W_constraint=maxnorm(1)
        # model.add(Dropout(0.2))
        model.add(Dense(7, activation='relu')) #,W_constraint=maxnorm(1)
        # model.add(Dropout(0.5))

        #output layer
        model.add(Dense(2, activation='softmax'))
        # model.summary()

    	# Compile model
        sgd = SGD(lr=learning_rate, momentum=0.7,nesterov=True)
        # adam=Adam(lr=learning_rate, beta_1=0.7, beta_2=0.999, epsilon=1e-08, decay=0.0000001)
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

        # Train the classifier on the training data and labels
        # history = model.fit(kX_train, kY_train, epochs=epochs, batch_size=batchSize, validation_split=0.2)
        history = model.fit(kX_train, kY_train,
                    batch_size=batchSize,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(kX_test, kY_test))

    	# Evaluate the model
    	# loss, accuracy = model.evaluate(kX_test, kY_test)
    	print "Using only ",dims ," Dimensions Accuracy: %.2f%%" % (np.amax(history.history['val_acc']))
    accuracies.append(np.amax(history.history['val_acc']));

for i in range(len(accuracies)):
	print '\t Using only ',i+1, ' we got accuracy of  ',accuracies[i]
print 'the best #of Dimensions to keep is ', 1+np.argmax(accuracies)

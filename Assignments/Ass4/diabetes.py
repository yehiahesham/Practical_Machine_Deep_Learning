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

def getAccuracy(trueLabels, predictions):
    correct = 0
    correct= np.sum(predictions == trueLabels)
    return (correct/float(len(trueLabels))) * 100.0



X_train, Y_train, mean, cov, feature_names = loadData('./data/pima-indians-diabetes.csv')

eigenvalues, eigenvectors = np.linalg.eig(cov);

print 'eigenvalues.shape is',eigenvalues.shape
print 'eigenvectors.shape is ',eigenvectors.shape # they are normalized already, IE each is a unit vector


# sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
batchSize = 32 #32
learning_rate=0.0001 #5.586261e-04   #0.0001
epochs=5 #200
num_folds = 5
dims_kept= [1,2,3,4,5,6,7,8]
dims_to_accuracies = {}
X_train_folds = []
y_train_folds = []

idxes = range(X_train.shape[0])
idx_folds = np.array_split(idxes, num_folds)

for dims in dims_kept:
    print '\t# Dimensions to Keep is ',dims
    newEigensIndices = np.argsort(-eigenvalues)[:dims]
    new_X_train = np.dot(X_train.as_matrix(),eigenvectors[newEigensIndices].T);
    print '\tnew_X_train.shape is ',new_X_train.shape
    # Define the classifier based on the # of Dimensions
    model = Sequential()
    model.add(Dense(128, input_shape=(dims,)))
    #model.add(Dropout(0.2))

    #hidden layers
    # model.add(Dense(64, activation='relu')) #,W_constraint=maxnorm(1)
    model.add(Dropout(0.2))

    model.add(Dense(32, activation='relu')) #,W_constraint=maxnorm(1)
    # model.add(Dropout(0.5))

    #output layer
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    # Train the classifier on the training data and labels
    sgd = SGD(lr=learning_rate, momentum=0.7,nesterov=True)
    # adam=Adam(lr=learning_rate, beta_1=0.7, beta_2=0.999, epsilon=1e-08, decay=0.0000001)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])


    for idxes in idx_folds:
        X_train_folds.append( new_X_train[idxes] )
        y_train_folds.append( Y_train[idxes] )

    dims_to_accuracies[dims] = list()
    for num in xrange(num_folds):
        print '\tProcessing fold', num, ' / ', num_folds

        X_cv_train = np.vstack( [ X_train_folds[x] for x in xrange(num_folds) if x != num ])
        y_cv_train = np.hstack( [ y_train_folds[x].T for x in xrange(num_folds) if x != num ])

        X_cv_test = X_train_folds[num]
        y_cv_test = y_train_folds[num]


        model.fit(X_cv_train, y_cv_train,
                  batch_size=batchSize,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(X_cv_test, y_cv_test))

        # Predict labels on the test data
        Y_predicated_test = model.evaluate(X_cv_test, y_cv_test, verbose=0)

        print 'Y_predicated_test is '
        print Y_predicated_test
        # print 'Y_predicated_test.shape is ',Y_predicated_test.shape
        # print Y_predicated_test

        # Compute and print the fraction of correctly predicted examples
        # dims_to_accuracies[dims].append(getAccuracy(y_cv_test,Y_predicated_test))
        # print ('Accurcy is ', dims_to_accuracies[k])


# for dims in dims_kept:
#     print '# Dimensions to Keep is ',dims

    # dims_to_accuracies[dims] = list()
    # for num in xrange(num_folds):
    #     print 'Processing fold', num, ' / ', num_folds
    #
    #     X_cv_train = np.vstack( [ X_train_folds[x] for x in xrange(num_folds) if x != num ])
    #     y_cv_train = np.hstack( [ y_train_folds[x].T for x in xrange(num_folds) if x != num ])
    #
    #     X_cv_test = X_train_folds[num]
    #     y_cv_test = y_train_folds[num]
    #
    #
    #     # Define the classifier based on the # of Dimensions
    #     model = Sequential()
    #     model.add(Dense(128, input_shape=(dims,)))
    #     #model.add(Dropout(0.2))
    #
    #     #hidden layers
    #     # model.add(Dense(64, activation='relu')) #,W_constraint=maxnorm(1)
    #     model.add(Dropout(0.2))
    #
    #     model.add(Dense(32, activation='relu')) #,W_constraint=maxnorm(1)
    #     # model.add(Dropout(0.5))
    #
    #     #output layer
    #     model.add(Dense(2, activation='softmax'))
    #     model.summary()
    #
    #     # Train the classifier on the training data and labels
    #
    #     # sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    #     batchSize = 32 #32
    #     learning_rate=0.0001 #5.586261e-04   #0.0001
    #     epochs=5 #200
    #
    #     sgd = SGD(lr=learning_rate, momentum=0.7,nesterov=True)
    #     # adam=Adam(lr=learning_rate, beta_1=0.7, beta_2=0.999, epsilon=1e-08, decay=0.0000001)
    #     model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    #
    #     model.fit(X_cv_train, y_cv_train,
    #               batch_size=batchSize,
    #               epochs=epochs,
    #               verbose=1,
    #               validation_data=(X_cv_test, y_cv_test))
    #
    #     # Predict labels on the test data
    #     Y_predicated_test = model.evaluate(X_cv_test, y_cv_test, verbose=0)
    #
    #     print 'Y_predicated_test[0] is '
    #     print Y_predicated_test[0]
    #     print 'Y_predicated_test.shape is ',Y_predicated_test.shape
    #     print Y_predicated_test
    #
    #     # Compute and print the fraction of correctly predicted examples
    #     dims_to_accuracies[dims].append(getAccuracy(y_cv_test,Y_predicated_test))
    #     print ('Accurcy is ', dims_to_accuracies[k])

#calculating for each K the mean and Std of the Accurcies
accuracies_mean = np.array([np.mean(v) for k,v in sorted(dims_to_accuracies.items())])
accuracies_std = np.array([np.std(v) for k,v in sorted(dims_to_accuracies.items())])

#printing for each K the mean and Std of the Accurcies
for dims, Accurcies in dims_to_accuracies.iteritems():
    print 'dims = ', dims,' got an Accurcy mean of ',np.mean(Accurcies), 'and Std of ', np.std(Accurcies), '\n'

#choosing Highest Accurcy mean with the least Std, respecting the difference threshold of best 2 means
c= np.column_stack((accuracies_mean,accuracies_std))
best = np.argmax(c,axis=0)[0]
threshold=0.0 #1.3
best_close = [i for i, j in enumerate(accuracies_mean) if j + threshold >= c[best][0]] #retreive close means the best mean found
for i in (best_close):
        if(c[i][1]<=c[best][1]):
            best=i

#printing best K and its Accurcy mean and Std
print ('best when dims= ', dims_kept[best], ' Acc= ', c[best][0] , ' std= ', c[best][1], '\n')

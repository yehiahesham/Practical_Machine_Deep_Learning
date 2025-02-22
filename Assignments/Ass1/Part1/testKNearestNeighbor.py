import numpy as np
import pylab as pl
import KNearestNeighbor as KNN
from PIL import Image
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
from datetime import datetime, date, time
import os
import pickle


def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = pickle.load(f)
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y

def load_CIFAR10(ROOT):
  """ load all of cifar """
  xs = []
  ys = []
  for b in range(1,6):
    f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del X, Y
  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
  return Xtr, Ytr, Xte, Yte


def scatterplot2(scores,decision,marker1='o',marker2='^',label1='',label2='',transp=1.0):
    scores1 = scores[decision == 1]
    scores2 = scores[decision == 0]
    pl.scatter(scores1[:,0], scores1[:,1],edgecolors='face', marker=marker1, label=label1, c='g',alpha=transp)
    pl.scatter(scores2[:,0], scores2[:,1],edgecolors='face', marker=marker2, label=label2, c='r',alpha=transp)

def unpickle(file): #opens pickled file and return a dictionary
    fo = open(file, 'rb')
    dict = pickle.load(fo)
    fo.close()
    return dict

def convertGrayScale(img):
    print 'Gray Scale is ON'
    # x = np.dot(img[...,:3], [0.299, 0.587, 0.114])
    # showimages(x,Gray=True)
    return np.dot(img[...,:3], [0.299, 0.587, 0.114])

def showimages(imges,Gray=False):
    plt.show()
    for i in range(len(imges)):
        if(Gray==True):
            plt.imshow(imges[i],cmap = plt.get_cmap('gray'))
        else:
            plt.imshow(imges[i])
        plt.pause(1)



# Use unpickle to return dictionaries of the data contained in the patch file & Meta file

cifar10_dir= '../cifar-10-batches-py/'
X_train, Y_train, X_test, Y_test = load_CIFAR10(cifar10_dir)
print 'X_train shape is ', X_train.shape , ' and Y_train shape is ', Y_train.shape
print 'X_test shape is ', X_test.shape , ' and Y_test shape is ', Y_test.shape



# X_train,Y_train = load_CIFAR_batch('/home/cse492/Desktop/cifar-10-batches-py/data_batch_1')
meta =  unpickle('../cifar-10-batches-py/batches.meta')
Classes= meta['label_names']


# #masking

# num_training2 = 20
# mask = range(num_training2)
# X_train = X_train[mask]
# Y_train=Y_train[mask]


X_train=  np.reshape(X_train, (X_train.shape[0], -1))

#generate test points
# X_test,Y_test = load_CIFAR_batch('/home/cse492/Desktop/cifar-10-batches-py/data_batch_2')

#masking
# num_test = 5
# mask = range(num_test)
# Y_test = Y_test[mask]
# X_test = X_test[mask]



X_test=  np.reshape(X_test, (X_test.shape[0], -1))



# create a Nearest Neighbor classifier class

KNN = KNN.KNearestNeighbor()


# Cross Validation
with open("bigest train, k=1.txt", "a") as myfile:
    v= 'X_train shape is ', X_train.shape , ' and Y_train shape is ', Y_train.shape, '\n'
    print str(v)
    myfile.write(str(v))

    v= 'X_test shape is ', X_test.shape , ' and Y_test shape is ', Y_test.shape, '\n'
    print str(v)
    myfile.write(str(v))

    start=datetime.now()
    myfile.write("Starting At time is ")
    myfile.write(str(start.time))

    v= 'Starting At time is ', start, '\n'
    print str(v)
    myfile.write(str(v))

    num_folds = 5
    k_options=[1,3,5,7,9,11]
    k_to_accuracies = {}
    X_train_folds = []
    y_train_folds = []

    idxes = range(X_train.shape[0])
    idx_folds = np.array_split(idxes, num_folds)


    for idxes in idx_folds:
        X_train_folds.append( X_train[idxes] )
        y_train_folds.append( Y_train[idxes] )

    for k in k_options:
        v= 'when K= ',k, '\n'
        print str(v)
        myfile.write(str(v))


        k_to_accuracies[k] = list()
        for num in xrange(num_folds):
            v=  'processing fold', num, ' / ', num_folds, '\n'
            print str(v)
            myfile.write(str(v))

            X_cv_train = np.vstack( [ X_train_folds[x] for x in xrange(num_folds) if x != num ])
            y_cv_train = np.hstack( [ y_train_folds[x].T for x in xrange(num_folds) if x != num ])

            X_cv_test = X_train_folds[num]
            y_cv_test = y_train_folds[num]


            KNN.train(X_cv_train, y_cv_train) #train the classifier on the training data and labels
            Y_predicated_test = KNN.predict(X_cv_test,l='L1',k=k) # predict labels on the test data

            # Compute and print the fraction of correctly predicted examples
            k_to_accuracies[k].append(KNN.getAccuracy(y_cv_test,Y_predicated_test))
            v='Accurcy is ', k_to_accuracies[k]
            print str(v)
            myfile.write(str(v))


    #calculating for each K the mean and Std of the Accurcies
    accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
    accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])

    #printing for each K the mean and Std of the Accurcies
    for k, Accurcies in k_to_accuracies.iteritems():
        v= 'K= ', k,' has Accurcy mean of ',np.mean(Accurcies), 'and Std of ', np.std(Accurcies), '\n'
        print str(v)
        myfile.write(str(v))


    #choosing Highest Accurcy mean with the least Std, respecting the difference threshold of best 2 means
    c= np.column_stack((accuracies_mean,accuracies_std))
    best = np.argmax(c,axis=0)[0]
    threshold=1.3
    best_close = [i for i, j in enumerate(accuracies_mean) if j + threshold >= c[best][0]] #retreive close means the best mean found
    for i in (best_close):
            if(c[i][1]<=c[best][1]):
                best=i

    #printing best K and its Accurcy mean and Std
    v= 'best k= ', k_options[best], ' Acc= ', c[best][0] , ' std= ', c[best][1], '\n'
    print str(v)
    myfile.write(str(v))


    #Time used
    end=datetime.now()
    print 'Ending At time ', end
    myfile.write(str(end.time))
    delta = end-start
    mins = delta.total_seconds() / 60
    print 'Time used is ' ,  mins
    myfile.write("Time used is ")
    myfile.write(str(mins))


#After finding out the best K , We predict on the testing set and check the Accurcy
#best K found was K=1, L1, 50,000 colored images training set


KNN.train(X_train, Y_train) #train the classifier on the training data and labels
print 'X_train shape is ', X_train.shape , ' and Y_train shape is ', Y_train.shape, '\n'
print 'X_test shape is ', X_test.shape , ' and Y_test shape is ', Y_test.shape, '\n'

Y_predicated_test = KNN.predict(X_test,l='L1',k=1) # predict labels on the test data

Acc,Testing_Accuray = KNN.getAccuracy(Y_test,Y_predicated_test)
print "#correct = %d out of %d"  % (Acc , len(Y_predicated_test) )
print "Finished Testing with accuracy = ",Testing_Accuray


X=[]
for c in range(len(Classes)):
    X.append(np.sum(Y_predicated_test[i]==c  and  Y_test[i]==c for i in range(Y_test.shape[0])))

correctPerClass = np.array(X)

for c in range(len(Classes)):
    print 'Got ', correctPerClass[c] , ' on class ',Classes[c], ' which is ', correctPerClass[c]/1000.0

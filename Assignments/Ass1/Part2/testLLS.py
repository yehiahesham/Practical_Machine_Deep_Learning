import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D
import os
import pickle
import LinearLeastSquare as LLS


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



# Use unpickle & load_CIFAR10 to return dictionaries of the data contained in the patch file & Meta file

cifar10_dir= '../cifar-10-batches-py/'
X_train, Y_train, X_test, Y_test = load_CIFAR10(cifar10_dir)

meta =  unpickle('../cifar-10-batches-py/batches.meta')
Classes= meta['label_names']


# masking

# num_training = 10
# mask = range(num_training)
# X_train = X_train[mask]
# Y_train=Y_train[mask]

#Convert to Gray Scale
# X_train = convertGrayScale(X_train)


#masking
# num_test = 5
# mask = range(num_test)
# Y_test = Y_test[mask]
# X_test = X_test[mask]

#Convert to Gray Scale
# X_test = convertGrayScale(X_test)


X_train=  np.reshape(X_train, (X_train.shape[0], -1))
X_test=  np.reshape(X_test, (X_test.shape[0], -1))

# create the true vector for each class for all images
X=[]
for c in range(len(Classes)):
    row = np.zeros((X_train.shape[0]),dtype=int)
    for i in range(len(Y_train)):
        if (Y_train[i] == c):
            row[i]=1
    X.append(row)

classesindexs = np.array(X)
print 'classesindexs is of shape ',classesindexs.shape

#add the ariftical 1 at the end of each image
X_train = np.c_[X_train, np.ones(X_train.shape[0])]
X_test = np.c_[X_test, np.ones(X_test.shape[0])]
print 'X_train shape is ', X_train.shape , ' and Y_train shape is ', Y_train.shape
print 'X_test shape is ', X_test.shape , ' and Y_test shape is ', Y_test.shape


# create a LinearLeastSquare classifier class
# LLS = LLS.LinearLeastSquare() # NOT Optimized
LLS = LLS.LinearLeastSquare(X_train) #Optimized


# calculate weights
X=[]
for c in range(len(Classes)):
    row=[]
    print 'getting weights of class ', Classes[c]
#     row = LLS.getweights(X_train,classesindexs[c]) # NOT Optimized
    row = LLS.getweights(classesindexs[c]) #Optimized
    X.append(row)

classesWeights = np.array(X)

print 'classesWeights is of shape ',classesWeights.shape
# print  classesWeights

Y_predicated_test = np.zeros(X_test.shape[0], dtype = Y_train.dtype)

print classesWeights.shape
print X_test.shape

for i in xrange(X_test.shape[0]):
    Scores=np.dot(classesWeights,X_test[i])
    max_index = np.argmax(Scores) # get the index with max score
    Y_predicated_test[i]=max_index

correct,Acc=LLS.getAccuracy(Y_test,Y_predicated_test)

print "#correct = %d out of %d"  % (correct , len(Y_test) )
print 'Accuracy is :' , Acc

X=[]
for c in range(len(Classes)):
    np.savetxt(Classes[c], classesWeights[c][:])
    X.append(np.sum( Y_predicated_test[i]==c  and  Y_test[i]==c for i in range(Y_test.shape[0])))

correctPerClass = np.array(X)

for c in range(len(Classes)):
    print 'Got ', correctPerClass[c] , ' on class ',Classes[c], ' which is ', correctPerClass[c]/1000.0

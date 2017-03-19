import pylab as pl
import numpy as np
import os
import pickle
from scipy.misc import imread
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D

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

def scatterplot2(scores,decision,marker1='o',marker2='^',label1='',label2='',transp=1.0):
    scores1 = scores[decision == 1]
    scores2 = scores[decision == 0]
    pl.scatter(scores1[:,0], scores1[:,1],edgecolors='face', marker=marker1, label=label1, c='g',alpha=transp)
    pl.scatter(scores2[:,0], scores2[:,1],edgecolors='face', marker=marker2, label=label2, c='r',alpha=transp)

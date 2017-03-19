import numpy as np
import math
from Layer import layer
from data_utils import *


def relu(x):
    # return max(0, x)
    return np.maximum(0, x)


def d_relu(output):
    #1 if output>0 , else 0
    # return (output!=np.array0)
    temp = 1.0 * (output > 0)
    return temp

def sigmod(x):
    return (1.0/(1.0+np.exp(np.multiply(-1,x) )))

def d_sigmod(output):
    return np.multiply(output,(1-output))

def updateWeight(dloss_dweight,oldweight,learningRate=0.1):
    return oldweight-(learningRate*dloss_dweight)

def softMax(x):
    exp = np.exp(x)
    scores = exp/np.sum(exp, axis=0, keepdims=True)
    return scores

def d_softMax(x):
    return x


def getsoftMax_loss_derror(x, expected):
    num_examples = x.shape[0]
    print x[2,0]
    print x[2,1]
    correct_probability = -np.log(x[range(num_examples), expected])
    print '1'
    data_loss = np.sum(correct_probability)
    data_loss = data_loss / num_examples

    # derivative of the activation SoftMax
    d_error = x
    d_error[range(num_examples), expected] -= 1
    d_error /= num_examples
    return data_loss,d_error


# testing_weights2=np.array([0.8,0.2])
# testing_weights3=np.array([0.4,0.9])
# testing_weights4=np.array([0.3,0.5])
# testing_weights5=np.array([0.3,0.5,0.9])
# testing_inputs=np.array([[1,1],[1,0],[0,1],[0,0]])
# exp=[0,1,1,0]
# testing_bias_layer1=np.array([0,0,0])
# testing_weights_layer1=np.array([0.8,0.4,0.3,0.2,0.9,0.5])
# testing_weights_layer1=np.reshape(testing_weights_layer1,(2,3))
#
# testing_bias_layer2=np.array([0])
# testing_weights_layer2=np.array([0.3,0.5,0.9])
# testing_weights_layer2=np.reshape(testing_weights_layer2,(3,1))

cifar10_dir= '/home/yehia/Desktop/ML/cifar-10-batches-py/'
X_train, Y_train, X_test, Y_test = load_CIFAR10(cifar10_dir)
print 'X_train shape is ', X_train.shape , ' and Y_train shape is ', Y_train.shape
print 'X_test shape is ', X_test.shape , ' and Y_test shape is ', Y_test.shape

meta =  unpickle('/home/yehia/Desktop/ML/cifar-10-batches-py/batches.meta')
Classes= meta['label_names']


#masking

num_training = 10
mask = range(num_training)
X_train = X_train[mask]
Y_train=Y_train[mask]

X_test=  np.reshape(X_test, (X_test.shape[0], -1))


learningRate=0.03
epochs=20
miniBatch=2
# numof_nodes,input_size,output_size,actFun,d_actFun,weights=None,biase=None)
# layer1=layer(3,2,relu,d_relu,weights=testing_weights_layer1,bias=testing_bias_layer1)
# layer2=layer(1,3,softMax,d_softMax,weights=testing_weights_layer2,bias=testing_bias_layer2)

print 'X_train shape is ', X_train.shape , ' and Y_train shape is ', Y_train.shape
print 'X_test shape is ', X_test.shape , ' and Y_test shape is ', Y_test.shape


layer1=layer(3,2,relu,d_relu)
layer2=layer(1,3,softMax,d_softMax)

for epoch in range (epochs):
    print 'epoch ',epoch
    for i in range (x_train.shape[0]/miniBatch):
        predicted= layer2.forward(layer1.forward(x_train[i*miniBatch:miniBatch*(i+1)]))
        print Y_test[i*miniBatch:miniBatch*(i+1)]
        data_loss,d_error = getsoftMax_loss_derror(predicted,Y_test[i*miniBatch:miniBatch*(i+1)])
        print '---1'

        # print 'predicted is ',predicted
        print 'data_loss is ',data_loss
        # print 'd_error is ',d_error
        z= layer2.backward(d_error)
        # print 'layer2.inputs.shape is ', layer2.inputs.shape
        # print 'z.shape is ',z.shape

        dw2=np.dot(layer2.inputs.transpose(),z)
        # print 'this is d_error/dweights of layer 2'
        # print  'dw2 is ',dw2
        # print  'dw2.shape is ',dw2.shape

        # print 'new updated of layer2'
        layer2.weights=updateWeight(dw2,layer2.weights,0.1)
        # print layer2.weights

        # print 'layer2.weights.shape is ',layer2.weights.shape
        # print 'z.shape is ',z.shape

        z= np.dot(z,layer2.weights.transpose())

        z= layer1.backward(z)

        # print 'this is d_error/dweights of layer 1'
        dw1= np.dot(layer1.inputs.transpose(),z)
        # print dw1

        # print 'new updated weights of layer1'
        layer1.weights=updateWeight(dw1,layer1.weights,0.1)
        # print layer1.weights

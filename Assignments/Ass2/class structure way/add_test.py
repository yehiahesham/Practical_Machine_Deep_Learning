from PIL import Image
import numpy as np
import math
from neuronV2 import neuron
import data_utils



def sigmod(x):
    return (1.0/(1.0+math.exp(-x)))

def d_sigmod(output):
    return output*(1-output)

def funnyf(x):
    return x*2

def d_funnyf(output):
    return 2

def updateWeight(dloss_dweight,oldweight,learningRate=0.1):
  return oldweight-(learningRate*dloss_dweight)

# v= sigmod(1.235)
# print v
# v2= d_sigmod(v)
# print v2 * -1* v *0.6899
#
# print ''
# v2= d_sigmod(v,v2)



testing_weights2=np.array([-0.754,0.066])

# testing_weights3=np.array([2.0,-3.0])
testing_biase= 0
testing_inputs=np.array([1,1])
# testing_input_size=2
# testing_output_size=1

# layer1=layer(2,3,sigmod,d_sigmod)
# layer2=layer(3*2,3,sigmod,d_sigmod)
# layer2=layer(3,1,sigmod,d_sigmod)
# layer3.forward(layer2.forward( layer1.forward(testing_inputs)))
# b = layer1.backward(layer2.backward(layer3.backward(1)))


# neuron=neuron(sigmod,d_sigmod,weights=testing_weights2,biase=testing_biase)
# print neuron.forward(testing_inputs)
# print neuron.backward([1.0])

# neuron=neuron(sigmod,d_sigmod,weights=testing_weights2,biase=testing_biase)

neuron5=neuron(sigmod,d_sigmod,weights=testing_weights2,biase=testing_biase)

predicted=1
learningRate=0.1
i=1
#testing AND, OR, XOR
while (predicted >= 0.1):
    print '--------------------------------------------------------------------------------'
    print 'iteration ',i
    i=i+1
    predicted= neuron5.forward(testing_inputs)
    print 'predicted is ' , predicted
    #output layer
    expected_output=1
    error=expected_output-predicted
    print 'error is ', error
    z5=(neuron5.backward([error]))

    #input layer
    print 'finially the dError/dwieghts in the first layer is : '
    dw11=z5*neuron5.inputs[0]
    dw12=z5*neuron5.inputs[1]

    print 'dw11 ------->',dw11
    print 'dw12 ------->',dw12

    #updating weights
    new_w11 = updateWeight(dw11,testing_weights2[0],learningRate)
    new_w12 = updateWeight(dw12,testing_weights2[1],learningRate)

    neuron5.weights=np.array([new_w11,new_w12])

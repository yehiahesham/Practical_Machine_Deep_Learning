from PIL import Image
import numpy as np
import math
from neuronV2 import neuron
import data_utils




def relu(x):
    return max(0, x)
    #return (1.0/(1.0+math.exp(-x)))

def d_relu(output):
    # return max(0, output)
    return (output!=0)
    # if output <= 0:
    #     return 0
    # else:
    #     return 1

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


testing_weights2=np.array([0.8,0.2])
testing_weights3=np.array([0.4,0.9])
testing_weights4=np.array([0.3,0.5])
testing_weights5=np.array([0.3,0.5,0.9])
testing_weights6=np.array([2.0,-3.0])
inputs=np.loadtxt('/home/yehia/Desktop/ML/Ass2/xor.txt')
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

neuron2=neuron(relu,d_relu,input_size=2,output_size=1)
neuron3=neuron(relu,d_relu,input_size=2,output_size=1)
neuron4=neuron(relu,d_relu,input_size=2,output_size=1)
neuron5=neuron(sigmod,d_sigmod,input_size=3,output_size=1)
# neuron6=neuron(sigmod,d_sigmod,weights=testing_weights6,biase=testing_biase)

learningRate=0.03
epochs=100

# testing XOR
# while (predicted >= 0.1):
for epoch in range (epochs):
    for i in range (inputs.shape[0]):
    #   print '--------------------------------------------------------------------------------'
    #   print 'iteration ',i
      expected_output= 2 - inputs[i][2]
      if expected_output == 0:
          expected_output = 0.15
      else:
          expected_output = 0.85

      predicted= neuron5.forward([neuron2.forward([inputs[i][0],inputs[i][1]]),
      neuron3.forward([inputs[i][0],inputs[i][1]]),
      neuron4.forward([inputs[i][0],inputs[i][1]])])
      i=i+1
      #output layer
      loss=0.5*(predicted-expected_output)*(predicted-expected_output)
      error=predicted-expected_output
      # print 'expected is', expected_output
    #   print 'predicted is ' , predicted
    #   print 'error is ', error
      print 'loss is ', loss
      z5=(neuron5.backward([error]))
      # print 'dError/dw(0.3) -----> ',z5*neuron2.output
      # print 'dError/dw(0.5) -----> ',z5*neuron3.output
      # print 'dError/dw(0.9) -----> ',z5*neuron4.output

      dw21=z5*neuron2.output
      dw22=z5*neuron3.output
      dw23=z5*neuron4.output


      # print ("----------------------------neuron5----------------------------------------")
      # print 'dError/d input_a(', neuron5.weights[0] , ') -----> ',z5* neuron5.weights[0]
      # print 'dError/d input_b(', neuron5.weights[1] , ') -----> ',z5* neuron5.weights[1]
      # print 'dError/d input_c(', neuron5.weights[2] , ') -----> ',z5* neuron5.weights[2]


      #hidden layer
      z2= neuron2.backward([z5* neuron5.weights[0]])
      # print ("----------------------------neuron2--------------------------------------------")
      z3= neuron3.backward([z5* neuron5.weights[1]])
      # print ("----------------------------neuron3--------------------------------------------")
      z4= neuron4.backward([z5* neuron5.weights[2]])
      # print ("----------------------------neuron4--------------------------------------------")

      #input layer
      # print 'finially the dError/dwieghts in the first layer is : '
      dw11=z2*neuron2.inputs[0]
      dw12=z2*neuron2.inputs[1]

      dw13=z3*neuron3.inputs[0]
      dw14=z3*neuron3.inputs[1]

      dw15=z4*neuron4.inputs[0]
      dw16=z4*neuron4.inputs[1]

      # print 'dw11 is = ',dw11
      # print 'dw12 is = ',dw12
      # print 'dw13 is = ',dw13
      # print 'dw14 is = ',dw14
      # print 'dw15 is = ',dw15
      # print 'dw16 is = ',dw16
      # print neuron3.backward([z1[1]])
      # print neuron4.backward([z1[2]])

      #updating weights
      # print 'old w11 is ', neuron2.weights[0]
      # print 'dw11  is ', dw11
      new_w11 = updateWeight(dw11,neuron2.weights[0],learningRate)
      new_w12 = updateWeight(dw12,neuron2.weights[1],learningRate)
      new_w13 = updateWeight(dw13,neuron3.weights[0],learningRate)
      new_w14 = updateWeight(dw14,neuron3.weights[1],learningRate)
      new_w15 = updateWeight(dw15,neuron4.weights[0],learningRate)
      new_w16 = updateWeight(dw16,neuron4.weights[1],learningRate)
      new_w21 = updateWeight(dw21,neuron5.weights[0],learningRate)
      new_w22 = updateWeight(dw22,neuron5.weights[1],learningRate)
      new_w23 = updateWeight(dw23,neuron5.weights[2],learningRate)
      # print 'bew w11 is ', new_w11


      neuron2.weights=np.array([new_w11,new_w12])
      neuron3.weights=np.array([new_w13,new_w14])
      neuron4.weights=np.array([new_w15,new_w16])
      neuron5.weights=np.array([new_w21,new_w22,new_w23])

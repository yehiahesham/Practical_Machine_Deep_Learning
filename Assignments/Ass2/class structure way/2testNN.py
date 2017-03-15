import numpy as np
import math
from neuronV2 import neuron
from layerV2 import layer
# import data_utils


def relu(x):
    return max(0, x)
    #return (1.0/(1.0+math.exp(-x)))

def d_relu(output):
    #1 if output>0 , else 0
    return (output!=0)

def sigmod(x):
    return (1.0/(1.0+math.exp(-x)))

def d_sigmod(output):
    return output*(1-output)


def updateWeight(dloss_dweight,oldweight,learningRate=0.1):
    return oldweight-(learningRate*dloss_dweight)


testing_weights2=np.array([0.8,0.2])
testing_weights3=np.array([0.4,0.9])
testing_weights4=np.array([0.3,0.5])
testing_weights5=np.array([0.3,0.5,0.9])

testing_weights_layer1=np.array([[0.8,0.2],[0.4,0.9],[0.3,0.5]])
testing_weights_layer2=np.array([[0.3,0.5,0.9]])
testing_biase= 0
testing_biase1= np.array([0,0,0])
testing_biase2= np.array([0])

testing_inputs=np.array([1,1])

# inputs=np.loadtxt('/home/yehia/Desktop/ML/Ass2/xor.txt')


learningRate=0.03
epochs=1


layer1=layer(3,2,3,relu,d_relu,weights=testing_weights_layer1,biase=testing_biase1)
layer2=layer(1,3,1,sigmod,d_sigmod,weights=testing_weights_layer2,biase=testing_biase2)

# v= sigmod(1.67)
# print  v
# v= sigmod(v)
# print  v



# testing XOR
for epoch in range (epochs):
    # for i in range (inputs.shape[0]):
#   print '--------------------------------------------------------------------------------'
#   print 'iteration ',i
#   expected_output= 2 - inputs[i][2]
  expected_output=0
#   if expected_output == 0:
#       expected_output = 0.15
#   else:
#       expected_output = 0.85

#   predicted= neuron5.forward([neuron2.forward([inputs[i][0],inputs[i][1]]),
#   neuron3.forward([inputs[i][0],inputs[i][1]]),
#   neuron4.forward([inputs[i][0],inputs[i][1]])])

  predicted = layer2.forward(layer1.forward(testing_inputs))
  # i=i+1

  #output layer
  loss=0.5*(predicted[0]-expected_output)*(predicted[0]-expected_output)
  error=predicted[0]-expected_output
  print 'predicted is ' , predicted
  print 'expected is', expected_output
  print 'error is ', error
  print 'loss is ', loss
  l2_dz=layer2.backward([error])
  print l2_dz

    #   print 'layer2.inputs is ',layer2.inputs
      #
    #   derro_dw_layer2       =l2_dz*layer2.inputs
    #   print 'dError/d weights (layer2) -----> ',derro_dw_layer2
      #
      #
    #   derro_dinputs_layer2  =l2_dz*layer2.weights
    #   print 'dError/d input (layer2) -----> ',derro_dinputs_layer2
      #
      #
    #   #hidden layer
    #   z2= neuron2.backward([l2_dz* neuron5.weights[0]])
    #   # print ("----------------------------neuron2--------------------------------------------")
    #   z3= neuron3.backward([l2_dz* neuron5.weights[1]])
    #   # print ("----------------------------neuron3--------------------------------------------")
    #   z4= neuron4.backward([l2_dz* neuron5.weights[2]])
    #   # print ("----------------------------neuron4--------------------------------------------")
      #
    #   #input layer
    #   # print 'finially the dError/dwieghts in the first layer is : '
    #   dw11=z2*neuron2.inputs[0]
    #   dw12=z2*neuron2.inputs[1]
      #
    #   dw13=z3*neuron3.inputs[0]
    #   dw14=z3*neuron3.inputs[1]
      #
    #   dw15=z4*neuron4.inputs[0]
    #   dw16=z4*neuron4.inputs[1]
      #
    #   # print 'dw11 is = ',dw11
    #   # print 'dw12 is = ',dw12
    #   # print 'dw13 is = ',dw13
    #   # print 'dw14 is = ',dw14
    #   # print 'dw15 is = ',dw15
    #   # print 'dw16 is = ',dw16
    #   # print neuron3.backward([z1[1]])
    #   # print neuron4.backward([z1[2]])
      #
    #   #updating weights
    #   # print 'old w11 is ', neuron2.weights[0]
    #   # print 'dw11  is ', dw11
    #   new_w11 = updateWeight(dw11,neuron2.weights[0],learningRate)
    #   new_w12 = updateWeight(dw12,neuron2.weights[1],learningRate)
    #   new_w13 = updateWeight(dw13,neuron3.weights[0],learningRate)
    #   new_w14 = updateWeight(dw14,neuron3.weights[1],learningRate)
    #   new_w15 = updateWeight(dw15,neuron4.weights[0],learningRate)
    #   new_w16 = updateWeight(dw16,neuron4.weights[1],learningRate)
    #   new_w21 = updateWeight(dw21,neuron5.weights[0],learningRate)
    #   new_w22 = updateWeight(dw22,neuron5.weights[1],learningRate)
    #   new_w23 = updateWeight(dw23,neuron5.weights[2],learningRate)
    #   # print 'bew w11 is ', new_w11
      #
      #
    #   neuron2.weights=np.array([new_w11,new_w12])
    #   neuron3.weights=np.array([new_w13,new_w14])
    #   neuron4.weights=np.array([new_w15,new_w16])
    #   neuron5.weights=np.array([new_w21,new_w22,new_w23])

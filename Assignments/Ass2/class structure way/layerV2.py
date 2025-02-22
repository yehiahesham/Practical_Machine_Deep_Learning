import numpy as np
from neuronV2 import neuron


class layer(object):
    """."""
    def __init__(self,numof_nodes,input_size,output_size,actFun,d_actFun,weights=None,biase=None):
        #weights intialization
        if weights is not None:
            self.weights=weights
        else: self.weights=np.random.randn(input_size, output_size) / np.sqrt(input_size)

        if biase is not None:
            self.biase=biase
        else: self.biase= np.random.randn(numof_nodes)

        self.numof_nodes=numof_nodes
        self.neuronsArray=[]
        self.input_size = input_size
        self.output_size = output_size

        # for i in range(len(self.weights)):
        for weight in self.weights:
            # neuronsArray.append(neuron(actFun,d_actFun,weights=self.weights,biase=self.biase))
            self.neuronsArray.append(neuron(actFun,d_actFun))


    def forward(self,inputs):
        self.inputs=inputs #to be used in backward propogation
        # print 'self.weights is ', self.weights
        # print 'inputs is ', inputs
        # print 'self.biase is ', self.biase
        self.outputs=[]
        for i in range(self.numof_nodes):
            self.outputs.append(self.neuronsArray[i].forward(inputs,self.weights[i],self.biase[i]))
        return self.outputs

    def backward(self,dz):
        nodes_dz=[]
        t=0
        # for i in range(self.output_size):
        #     for gradient in dz:
        #         print gradient * self.neuronsArray[i].backward(gradient)
        #         nodes_dz.append(self.neuronsArray[i].backward(dz[i]))

        # the returned dz is of size [#fo neurons]
        for i in range(self.numof_nodes):
            ???
            nodes_dz.append(self.neuronsArray[i].backward(dz[i]))
        print nodes_dz
        return nodes_dz

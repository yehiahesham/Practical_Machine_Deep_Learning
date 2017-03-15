import numpy as np
import gates
from toposort import toposort, toposort_flatten

class neuron(object):
    """docstring for neuron."""
    def __init__(self,actFun,d_actFun,weights=None,biase=None):
        #weights intialization
        self.gates=[]

        if weights is not None:
            self.weights=weights
        else: self.weights=np.random.randn(input_size, output_size) / np.sqrt(input_size)

        if biase is not None:
            self.biase=biase
        else: self.biase= np.zeros(output_size)

        #Activation function and its derivative are the same for all nodes in the layer
        self.actFun=actFun
        self.derivative_actFun=d_actFun

    def insertGate(self,gate):
        self.gates.append(gate)



    def forward(self,inputs):
        self.inputs=inputs #to be used in backward propogation
        self.outputs=self.actFun(np.dot(self.weights.transpose(),inputs)+self.biase)

        # print 'self.weights is ', self.weights
        # print 'inputs is ', inputs
        # print 'self.biase is ', self.biase
        # print 'W.T X = ',np.dot(self.weights.transpose(),inputs)
        # print 'acct(W.T X) = ',self.outputs

        for i in range(len(inputs)):
            self.insertGate(gates.Multiply(inputs[i],self.weights[i]))


        t=0
        for i in range(len(self.gates)-1):
            print i
            for j in range(i+1,len(self.gates)):
                 gates.Add(self.gates[i].forward(),self.gates[j].forward())
                 self.insertGate(t)
        print 'this is t ',t
        # return t

    def backward(self,dz):
        #backward of the acctivation function
        t=self.derivative_actFun(self.outputs)
        print 'derivative of the acc is ',t
        new_dz=dz*self.derivative_actFun(self.outputs)

        #cal backward of WT here
        new_dz=new_dz

        return new_dz

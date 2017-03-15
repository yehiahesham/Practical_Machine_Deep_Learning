import numpy as np

class layer(object):
    """."""
    def __init__(self,input_size,output_size,actFun,d_actFun,weights=None,biase=None):
        #weights intialization

        if weights is not None:
            self.weights=weights
        else: self.weights=np.random.randn(input_size, output_size) / np.sqrt(input_size)

        if biase is not None:
            self.biase=biase
        else: self.biase= np.zeros(output_size)

        # self.numof_nodes=numof_nodes

        #Activation function and its derivative are the same for all nodes in the layer
        self.actFun=actFun
        self.derivative_actFun=d_actFun

    def forward(self,inputs):
        self.inputs=inputs #to be used in backward propogation
        self.outputs=self.actFun(np.dot(self.weights.transpose(),inputs)+self.biase)
        return self.outputs

    def backward(self,dz):
        #backward of the acctivation function
        t=self.derivative_actFun(self.outputs)
        print 'derivative of the acc is ',t
        new_dz=dz*self.derivative_actFun(self.outputs)

        #cal backward of WT here
        new_dz=new_dz

        return new_dz

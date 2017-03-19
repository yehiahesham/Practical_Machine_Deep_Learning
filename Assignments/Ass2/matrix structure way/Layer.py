import numpy as np

class layer(object):
    """."""
    def __init__(self,output_size,input_size,actFun,d_actFun,weights=None,bias=None):
        #weights intialization

        if weights is not None:
            self.weights=weights
        else: self.weights=np.random.randn(input_size, output_size) / np.sqrt(input_size)

        if bias is not None:
            self.bias=bias
        else: self.bias= np.random.randn(1,output_size) / np.sqrt(input_size)


        #Activation function and its derivative are the same for all nodes in the layer
        
        self.actFun=actFun
        self.derivative_actFun=d_actFun

    def forward(self,inputs):
        self.inputs=inputs #to be used in backward propogation  
        self.outputs=self.actFun(np.dot(inputs,self.weights)+self.bias)
        return self.outputs

    def backward(self,dz):
        return self.derivative_actFun(dz)
        #backward of the acctivation function

import numpy as np

class neuron(object):
    """docstring for neuron."""
    def __init__(self,actFun,d_actFun,input_size=None, output_size=None,weights=None,biase=None):
        #weights intialization

        if weights is not None:
            self.weights=weights
        else: self.weights=np.random.randn(input_size, output_size) / np.sqrt(input_size)

        if biase is not None:
            self.biase=biase
        else: self.biase= np.zeros(output_size)



        #Activation function and its derivative are the same for all nodes in the layer
        self.actFun=actFun
        self.derivative_actFun=d_actFun

    def forward(self,inputs):
        self.inputs=inputs #to be used in backward propogation
        self.output=self.actFun(np.sum(inputs* self.weights)+self.biase)
        return self.output

    def backward(self,dz):
        new_dz=[]
        cumulative_dz=0

        #adding multiple gradients to the same neuron
        for neuron_gradients in dz:
            cumulative_dz=cumulative_dz+neuron_gradients

        # getting the derivative of the acctivation function
        # print 'cumulative_dz -------> ',cumulative_dz
        # print 'derivative_actFun(output), where output = actFun(SIGMA) -------> ',self.derivative_actFun(self.output)
        cumulative_dz=cumulative_dz*self.derivative_actFun(self.output)
        # print ' now z passed the sigmod in the backward, remaing WX, current z is  ------> ',cumulative_dz
        return cumulative_dz

        #calculating the of each dz/dw
        # print 'z * each input  '
        # for neuron_input in self.inputs:
        #     print neuron_input,' * z(',cumulative_dz,') ====>', cumulative_dz*neuron_input
        #     new_dz.append(cumulative_dz*neuron_input)
        #
        # # print 'new_dz -------> ',new_dz
        # return new_dz

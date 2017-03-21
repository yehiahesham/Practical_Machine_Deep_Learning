import numpy as np
import Layer as layer


class NeuralNetwork:

    def __init__(self, layer_num,layer_node_description ):
        # layer_node_description is the number of nodes & its inputs
        self.layers = []
        for i in range (layer_num):
            self.addLayer(layer_node_description[i][0],layer_node_description[i][1])
            
    def addLayer(self, num_neuron,input_size):
        self.layers.append(layer(num_neuron,input_size))


    def train(self, x_train, y_train, x_valid, y_valid, epochs, mini_batch=10, lr=0.1):
        # loss = 0
        # reg_loss = 0
        # x_train = self.pre_process_data(x_train)
        # x_valid = self.pre_process_data(x_valid)

        for epoch in range (epochs):
            print 'epoch ',epoch
            for i in range (x_train.shape[0]/miniBatch):
                batch_num = j*mini_batch
                x = x_train[i*miniBatch:miniBatch*(i+1)]
                y = y_train[i*miniBatch:miniBatch*(i+1)]

                self.calForward(x)
                loss, reg_loss = self.backward(y)
                self.update_weights(lr)

    @staticmethod
    def pre_process_data(x_data):
        x_data -= np.mean(x_data, axis=0)
        x_data /= np.std(x_data, axis=0)

        return x_data


    def update_weights(self, lr):
        return oldweight-(learningRate*dloss_dweight)

    def calForward(self, x_train):
        #create the layers here first
          self.addLayer(x_train,)
         # self.addLayer(outputs_num,)
        # first layer, the input layer
        self.layers[0].inputs = x_train

        for i in xrange(1, len(self.layers)-1):
            w = self.layers[i-1].weights
            b = self.layers[i-1].bias
            a = self.layers[i-1].a

            result = layer.NeuralLayer.compute(w, a, b)
            result = layer.NeuralLayer.activate_relu(result)
            self.layers[i].a = result

        w = self.layers[-2].weights
        b = self.layers[-2].bias
        a = self.layers[-2].a

        result = layer.NeuralLayer.compute(w, a, b)
        result = layer.NeuralLayer.activate_soft_max(result)
        self.layers[-1].a = result

        return result

    def backward(self, y, reg=1e-3):
        # From hidden to output layer
        o = self.layers[-1].a
        delta, loss = self.calculate_loss_soft_max(o, y)
        reg_loss = self.calculate_reg_loss()

        a = self.layers[len(self.layers)-2].a

        dl_dw = np.dot(a.transpose(), delta)
        dl_dw += reg*self.layers[len(self.layers)-2].weights
        dl_db = np.sum(delta, axis=0, keepdims=True)

        self.layers[-1].delta = delta
        self.layers[len(self.layers)-2].dl_dw = dl_dw
        self.layers[len(self.layers)-2].dl_db = dl_db
        ################################

        # input to hl, or hl to hl without bias
        for i in reversed(xrange(len(self.layers)-2)):
            weights = self.layers[i+1].weights
            a = self.layers[i+1].a

            delta = np.dot(delta, weights.transpose())
            activation_prime = layer.NeuralLayer.diff_relu(a)
            delta = np.multiply(activation_prime, delta)

            x = self.layers[i].a
            dl_dw = np.dot(x.transpose(), delta)
            dl_dw += reg * self.layers[i].weights
            dl_db = np.sum(delta, axis=0, keepdims=True)

            self.layers[i].dl_dw = dl_dw
            self.layers[i].dl_db = dl_db
            self.layers[i+1].delta = delta
        ##################################

        return loss, reg_loss

    def calculate_reg_loss(self, reg=1e-3):
        reg_loss = 0
        for i in xrange(len(self.layers) - 1):
            w = self.layers[i].weights
            reg_loss += 0.5 * reg * np.sum(w * w)

        return reg_loss

    def predict(self, x_test, y_test):
        scores = self.calForward(x_test)
        predicted_class = np.argmax(scores, axis=1)
        accuracy = np.mean(predicted_class == y_test)
        print 'accuracy',accuracy

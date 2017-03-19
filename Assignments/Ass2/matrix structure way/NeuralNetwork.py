import numpy as np
import Layer as layer


class NeuralNetwork:

    def __init__(self, layer_num,layer_node_description ):
        # layer_node_description is the number of nodes & its inputs
        self.layers = []
        for i in range (layer_num):
            self.addLayer(layer_node_description[i][0],layer_node_description[i][1])
        # self.addLayer(inputs_num)
        # self.addLayer(outputs_num,)

    def addLayer(self, num_neuron,input_size):
        self.layers.append(layer(num_neuron,input_size))


    def train(self, x_train, y_train, x_valid, y_valid, epochs, mini_batch=10, lr=0.1):
        # self.create_model()
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
                loss, reg_loss = self.propagate_back(y)
                self.update_weights(lr)

            # if epoch % 5 == 0:
            #     out_train = self.calForward(x_train)
            #     out_valid = self.calForward(x_valid)
            #     print 'epoch: ', epoch
            #     print '\tLoss: ', loss
            #     print '\tReg Loss: ', reg_loss
            #     print '\tAccuracy Train: ', self.get_accuracy(out_train, y_train)
            #     print '\tAccuracy Valid: ', self.get_accuracy(out_valid, y_valid)
            #     print '\n'

    @staticmethod
    def pre_process_data(x_data):
        x_data -= np.mean(x_data, axis=0)
        x_data /= np.std(x_data, axis=0)

        return x_data

    def create_model(self):
        self.layers += [self.layers.pop(1)]  # Move the output layer to be the last layer
        self.seed_weights()

    def seed_weights(self):
        for i in xrange(0, len(self.layers)-1):
            fan_in = self.layers[i].neurons  # +1 for the bias
            fan_out = self.layers[i+1].neurons
            w = np.random.randn(fan_in, fan_out)/np.sqrt(fan_in/2.0)
            # w = total_weights[0:fan_in-1,:]
            # b = total_weights[[-1]]
            b = np.random.randn(1, fan_out)/np.sqrt(1/2.0)

            self.layers[i].set_weights(w, b)

    def update_weights(self, lr):
        for i in xrange(len(self.layers)-1):
            dl_dw = self.layers[i].dl_dw
            dl_db = self.layers[i].dl_db

            delta_w = -lr*dl_dw
            delta_b = -lr*dl_db

            new_weights = self.layers[i].weights + delta_w
            new_bias = self.layers[i].bias + delta_b

            self.layers[i].set_weights(new_weights, new_bias)

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

    def propagate_back(self, y, reg=1e-3):
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

    @staticmethod
    def calculate_loss_soft_max(o, t):
        num_examples = o.shape[0]
        correct_probability = -np.log(o[range(num_examples), t])
        data_loss = np.sum(correct_probability)
        data_loss = data_loss / num_examples

        # Softmax prime
        d_error = o
        d_error[range(num_examples), t] -= 1
        d_error /= num_examples

        return d_error, data_loss

    def calculate_reg_loss(self, reg=1e-3):
        reg_loss = 0
        for i in xrange(len(self.layers) - 1):
            w = self.layers[i].weights
            reg_loss += 0.5 * reg * np.sum(w * w)

        return reg_loss

    def test(self, x_test, y_test):
        scores = self.calForward(x_test)
        print 'accuracy: %.2f' % (self.get_accuracy(scores, y_test))

    @staticmethod
    def get_accuracy(scores, y_test):
        predicted_class = np.argmax(scores, axis=1)
        accuracy = np.mean(predicted_class == y_test)

        return accuracy

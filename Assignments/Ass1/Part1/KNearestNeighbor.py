import numpy as np
import operator

class KNearestNeighbor(object):
    #http://cs231n.github.io/classification/
    def __init__(self):
        pass

    def train(self, X, y):
        """ X is N x D where each row is an example. Y is 1-dimension of size N """
        # the nearest neighbor classifier simply remembers all the training data
        self.Xtr = X
        self.ytr = y


    def predict(self, X, l='L1',k=3):
        """ X is N x D where each row is an example we wish to predict label for """
        num_test = X.shape[0]


        # lets make sure that the output type matches the input type
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

        # loop over all test rows
        for i in xrange(num_test):
            # print i
            # find the nearest training example to the i'th test example
            if l == 'L1':
                # using the L1 distance (sum of absolute value differences)
                distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
            else:
                distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis = 1))

            min_indexs=np.argpartition(distances,k)[:k] # get indices of the k smallest distances(3 is default)
            Votes=np.zeros(10, dtype=np.int)
            #get the vote count
            for vote in min_indexs:
                Votes[self.ytr[vote]] += 1
            max_index = np.argmax(Votes)
            Ypred[i] = max_index  # prediction based on nearest K  votes
        return Ypred

    def getAccuracy(self,trueLabels, predictions):
        correct = 0
        correct= np.sum(predictions == trueLabels)
        # print "#correct = %d out of %d"  % (correct , len(trueLabels) )
        return correct,(correct/float(len(trueLabels))) * 100.0

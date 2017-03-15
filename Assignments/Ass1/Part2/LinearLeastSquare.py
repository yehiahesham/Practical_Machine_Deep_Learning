import numpy as np

class LinearLeastSquare(object):
    def __init__(self,X= None):
        self.optimize=False
        if X is not None:
            self.optimize=True
            X_T = X.transpose()
            self.Const = np.dot(np.linalg.inv(np.dot(X_T,X)),X_T)
        
    def getweights(self,truelabels): 
            #optimizied
            if(self.optimize):
                return np.dot(self.Const,truelabels)
            else:
                print 'Error!, You should intialize with Trainning Data first to used the optomized getweights function '
                return 42 # error !

    def getweights_withX(self,X,truelabels):
            #NOT optimizied
            X_T = X.transpose()
            return np.dot(np.dot(np.linalg.inv(np.dot(X_T,X)),X_T),truelabels)
        
    def getAccuracy(self,trueLabels, predictions):
            correct = 0
            correct= np.sum(predictions == trueLabels)
            return correct,(correct/float(len(trueLabels))) * 100.0

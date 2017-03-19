# t=inputs[i*miniBatch:miniBatch*(i+1), 0]
# t2 = inputs[i*miniBatch:miniBatch*(i+1), 1]
# t = np.c_[t, t2]
# expected=inputs[i*miniBatch:miniBatch*(i+1),2]
# t=np.reshape(t,(miniBatch,2))
# print t
# print expected.shape

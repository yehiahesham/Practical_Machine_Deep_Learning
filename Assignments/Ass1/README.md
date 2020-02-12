# Practical_Machine_Deep_Learning

## Part 1 ##
Implement a k nearest neighbor (k-NN) classifier that can best recognize the
10 different classes in the CIFAR-10 dataset.

Details:


 * Download the dataset from http://www.cs.toronto.edu/~kriz/cifar.html The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.
 * Nested bullet
 * Train your best k-NN using the 50000 training set.
 * Using 5-crossfold validation technique, find the best “k” for this dataset. Answer also the following:
   * Is L1 distance better than L2 in this problem?
   * Is grayscale image space better than the original color space? 
   * Use the 10000 testing set to report the accuracy of your classifier.
   
## Part 2 ##

Implement a linear least square classifier (LLS) that can best recognize the
10 different classes in the CIFAR-10 dataset.

=======
### Calculating W for the LLS classified ###

For each classifier we are going to give it the 50,000; however, for each classifier we are going to label which index from the 50,000 are its specifically, which is actual the true label vector. The true label vector will be 1s for the desired class and 0s for everything else. Now,  for each classier , its weights are calculated as the following matrix formula:

                          Weigths = W = inverse of(X.T dot X) dot X.T dot Truelables
                          
and then you will have a (10x3073) Weights vector. Why 3037 ? because we have 32x32x3+1 features. And extra one is used for W0, which is the bias. Thus, a single testing image and training Image must be (3073x1) which we achieve b y adding an extra 1 at the end. Next, we apply the following matrix formula:

                          F(x,W)= Wx

Now,we can have 10 scores which we will take  the highest and assign that classer's label to the testing image as the a classification answer.
Now to test our LSS classifier against the testing set, I am going to have training set size of 50,000 colored images to report the accuracy of my classifier . The simulation gave:

### Average Correct Classification Rate (ACCR) : ### 

#correct = 3637 out of 10000
Accuracy is : 36.37

Correct Classification Rate of each of the 10 classes separately  (CCRn):
Got  469  on class  airplane  which is  0.469
Got  445  on class  automobile  which is  0.445
Got  207  on class  bird  which is  0.207
Got  177  on class  cat  which is  0.177  
Got  243  on class  deer  which is  0.243
Got  285  on class  dog  which is  0.285
Got  449  on class  frog  which is  0.449
Got  426  on class  horse  which is  0.426
Got  508  on class  ship  which is  0.508
Got  428  on class  truck  which is  0.428

Now to test our LSS classifier against the testing set, I am going to have training set size of 50,000 Gray Scale images to report the accuracy of my classifier . The simulation gave:

### Average Correct Classification Rate (ACCR) : ### 

#correct = 2672 out of 10000
Accuracy is : 26.72

### Correct Classification Rate of each of the 10 classes separately  (CCRn): ### 

Got  293  on class  airplane  which is  0.293
Got  358  on class  automobile  which is  0.358
Got  155  on class  bird  which is  0.155
Got  125  on class  cat  which is  0.125
Got  144  on class  deer  which is  0.144
Got  282  on class  dog  which is  0.282
Got  223  on class  frog  which is  0.223
Got  253  on class  horse  which is  0.253
Got  382  on class  ship  which is  0.382
Got  457  on class  truck  which is  0.457


I compared the ACCR  & CCRn from both colored and gray scale in LLS with 50,000. We can see that the colored images with produce better ACCR  & CCRn  than using Gray scaled images . So, Colored images are better, Yet it is faster at calculation.


### Report CCRn for k-NN versus LLS classifiers ### 


	                airplane automobile	bird	cat	  deer	dog	  frog	horse	ship	truck
            KKN	  0.523	    0.297	 0.407 	0.264	  0.452	0.306	0.371	0.343	0.62	0.276
            LLS	  0.469	    0.445	 0.207 	0.177	  0.243	0.285	0.449	0.426	0.508	0.428
            Winner    KKN	   LLS            KKN	  KKN	  KKN	  KKN	  LLS	  LLS	    KKN	LLS

Seems KNN is better in most of classes. Thus, I can say  KNN has better CCRn than that of LLS.

### Report ACCR for k-NN versus LLS classifiers ### 

The KNN had better ACCR than did LLS.

Over fit the data for either the k-NN or the LLS classifiers?

For the LLS we are using linear so that M=1. Moreover, for both LLS and KNN ,we are using the whole dataset as training, which should remove any over fitting if happen. Likewise we are testing on the whole testing set. Moreover, the testing error didn't go off and increased while the  training errors kept zero. If it happened, that would be a clear sign.

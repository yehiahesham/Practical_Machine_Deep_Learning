# Practical_Machine_Deep_Learning

Implement a classifier to differentiate between the two classes diabetes vs
no_diabetes. Compare your best accuracy with/without using PCA for
dimensionality reduction.


Details:


 * Download the dataset from
https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes
* The dataset consists of 768 data points in 2 classes. Each data point
consists of 8 features.
* Using 5-crossfold validation technique, train your best classifier using
all 8 features of the data.
* Repeat the previous after applying PCA to reduce the data dimensionality.

=============

Results:


* approximately 71 % accuracy. 
* PCA help in enhancing the classification accuracy by 6.9034886837%. That is because you simplified the input size and thus lowered # of total weights and thus reduce overfitting. As I am using 2 FC layers and an output layer. Likewise, your prediction & training now are faster as you need only one feature instead of 8.


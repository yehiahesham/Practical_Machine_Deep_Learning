# Practical_Machine_Deep_Learning

   # Description #

This is a miniature of ImageNet classification Challenge at Kaggle https://kaggle.com/join/tinyimagenetAUC .

MicroImageNet classification challenge is similar to the classification challenge in the full ImageNet ILSVRC. MicroImageNet contains 200 classes for training. Each class has 500 images. The test set contains 10,000 images. All images are 64x64 colored ones.

Your objective is to classify the 10,000 test set as accurately as possible.
Acknowledgements

We thank Fei-Fei Li, Andrej Karpathy, and Justin Johnson for providing this dataset as part of their cs231n course at Stanford university http://cs231n.stanford.edu/

   # File descriptions #

 * train.images.zip - the training set (images distributed into class labeled folders). You can download it from here : https://inclass.kaggle.com/c/tiny-imagenet/download/train.images.zip
 * test.zip - the unlabeled 10,000 test images
 * sample.txt - a sample submission file in the correct format (but needs to have 10,001 lines. One line per image in addition to the first header line)
 * wnids.txt - list of the used ids from the original full set of ImageNet
 * words.txt - description of all ids of ImageNet




   # Evaluation #

  The evaluation metric for this competition is the average correct classification rate among all 200 classes. Submission files should contain two comma separated columns: ImageFileName, and Predicted class label.

  For submission Format check the sample.txt
  
  # What I Have done to get 82.62% ACCR #
When I started tackling the Assignment, I knew I wanted to have a pertained model to achieve a high accuracy that I don’t need to train from scratch. Digits was the easers thing to start with, no coding just dropdown boxes and etc. So I wanted to first run the AlexNet on our dataset, just as a proof of concept. But almost figured out that the training folder needed to be tuned in order to load it to Digits, as well as the validation set. So, I wrote two bash scripts to setup the training set and to extract the first 50 images from each class (represented as the folder name) to the validation folder under the same class name (folder name) . I got a very bad validation accuracy, then I understood that the problem was when I loaded the pertained weights of the model, I had to remove the weights on the last Fully Connected Layers in order to have my own output to only 200 classes and make the weights of the Convolutional layers learning rate small to make them non trainable, which is fine tuning the architecture. Now I had to try better architecture, like GoogleNet. So I searched for GoogleNet’s pertained model and weights and run it. A working GoogleNet, got me accuracy of 60.5% on only 10 epochs & no overfitting happened.  I kept improving and changing, like removing the regression part etc., on the model till I got accuracy of 70.5%  and 10 epochs  and no overfitting, with small learning rate to the model weights and trained the a newly added layer in the Fully Connected layers. 
  
I felt that this might be the best I could achieve with GoogleNet, thus I thought about Inception. At that point I found Keras better as it has the pertained model and weight so of InceptionV3. Thus I shifted to Keras. So I ran the InceptionV3 model and got accuracy of 0.77120, replaced the FC layers with just an output layer of 200 neurons obviously, no preprocessing on training set and no Reduce Learning rate on Plateau, which reduces the learning rate once the accuracy begin to saturate on in order to reach the best solution. 

Likewise with GoogleNet, that was the maximum I could reach with InceptionV3. So I tried to search for what is better, and found the Xception architecture on the keras documentation. So it was worth the shot. I downloaded the model and its weights, removed the last FC layers and put mine. Using Xception, I got accuracy of 82.040% with data generator’s normalization and augmentation, with reduced learning on Plateau. That was the highest I could achieve. However, I felt that may be over fitted a little bit as the training loss was lower than the validation loss between 0.3 to 0.4 differences. I wrote the code which use the Keras’s preprocessing for Xception the data in order to even have a better results but I was out of time. 

However in the end all what I kept doing is to find the best hyper parameters and tricks that should lead to better results, like training the FC layers alone while the Conv. Layers’ weights were fixed, then use those waits to train all layers together, small learning rate or bit big but with a reducing function, batch size which was necessary due to the memory.


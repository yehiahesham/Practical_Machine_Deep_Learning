from InceptionV3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.optimizers import SGD, Adam, Nadam, RMSprop
from imagenet_utils import preprocess_input, decode_predictions
from keras.models import Sequential
from keras.layers.core import Dense, Activation,Dropout ,Flatten
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras.constraints import maxnorm
from keras.models import load_model
from keras.callbacks import ModelCheckpoint,TensorBoard
import keras
from keras import backend as K
import numpy as np

img_size = 224
batchSize = 64 #32
learning_rate=0.01
epochs=0   #0 #20






model = InceptionV3(include_top=False, weights='imagenet', classes=200, pooling='avg', input_shape=(img_size, img_size, 3))

# for layer in model.layers:
#     layer.trainable = False

output_model = model.output
predictions = Dense(200, activation='softmax', name='predictions')(output_model)

new_model = Model(input=model.input, output=predictions)
new_model.summary()



#Loading best weights
new_model.load_weights("./weights/weights_1.hdf5")



sgd = SGD(lr=learning_rate, nesterov=True, momentum=0.7, decay=1e-4)
new_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

#--------------------------

datagen_test = ImageDataGenerator(
    # featurewise_center=True,
    # samplewise_center=False,
    # featurewise_std_normalization=True,
    # samplewise_std_normalization=False
    )

datagen_train = ImageDataGenerator(
    # featurewise_center=True,
    # samplewise_center=False,
    # featurewise_std_normalization=True,
    # samplewise_std_normalization=False,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

board = keras.callbacks.TensorBoard(log_dir='./logs/log_2', histogram_freq=0, write_graph=True, write_images=False)
checkpointer = ModelCheckpoint(filepath="./weights/weights_2.hdf5", verbose=1, save_best_only=True, monitor="val_loss")

train_gen_trainData  = datagen_train.flow_from_directory('/home/yehiahesham/Desktop/ML/Ass3_data/train/',
 target_size=(img_size, img_size),
 batch_size=batchSize,
 class_mode='categorical')


train_gen_testData = datagen_test.flow_from_directory('/home/yehiahesham/Desktop/ML/Ass3_data/validate/',
   target_size=(img_size, img_size),
   batch_size=batchSize,
   class_mode='categorical')


# history = new_model.fit_generator(train_gen_trainData, steps_per_epoch= 90000 / batchSize ,
#                             epochs=epochs,validation_data=train_gen_testData,validation_steps =10000 / batchSize ,verbose=1, callbacks=[board,checkpointer] )


#--------------------------


# predicted_classes = new_model.predict_classes(X_test, batch_size=1, verbose=1)
# preds = new_model.predict(x)

# print (predicted_classes)
# reshaped_y_test=np.reshape(y_test,(10000,))
# num_correct = np.sum(predicted_classes == reshaped_y_test)
#
# print (num_correct)
# acc = float(num_correct)/10000.0
# print(acc*100)




#-------------------------

# img_path = '/home/yehiahesham/Desktop/ML/Ass3_data/train/n07873807/n07873807_72.JPEG'
# img = image.load_img(img_path, target_size=(299, 299))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
#
# x = preprocess_input(x)
#
# preds = model.predict(x)
# print preds.shape
# print('Predicted:',
#--------------------------------


def preprocess_input(x):
    x/=255.
    x-=0.5
    x*=2.
    return x




l=[None]*200

for key,value in train_gen_trainData.class_indices.items():
    l[value]=key

f = open('Submission', 'w')
f.write('Id,Prediction\n')

for i in range(0, 10000):
    img_name = 'test_%d' % (i,)
    img_name += '.JPEG'
    img_path = '/home/yehiahesham/Desktop/ML/Ass3_data/test_images/' + img_name
    img = image.load_img(img_path, target_size=(img_size, img_size))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)
    preds = new_model.predict(x)
    f.write(img_name + ',' + l[np.argmax(preds)] + '\n')
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    # print('Predicted:', decode_predictions(preds, top=3)[0])
    # print()

f.close()
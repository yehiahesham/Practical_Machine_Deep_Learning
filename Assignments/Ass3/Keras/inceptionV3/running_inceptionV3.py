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
from keras.callbacks import ModelCheckpoint,TensorBoard, ReduceLROnPlateau
from keras.applications import Xception
import keras
from keras import backend as K
import numpy as np
import cv2
import os
import glob
import math

img_size = 299 #224 #299
batchSize = 16 #32
learning_rate=0.01
epochs=15   #0 #20


def get_im(path):
    # Load as NOT grayscale
    img = cv2.imread(path, 1)
    # Reduce size
    resized = cv2.resize(img, (img_size, img_size))
    return resized


def load_train():
    X_train = []
    y_train = []
    print('Read train images')
    for j in range(10):
        print('Load folder c{}'.format(j))
        path = os.path.join('..', 'input', 'imgs', 'train', 'c' + str(j), '*.jpg')
        files = glob.glob(path)
        for fl in files:
            img = get_im(fl)
            X_train.append(img)
            y_train.append(j)

    return X_train, y_train


def load_test():
    print('Read test images')
    path = os.path.join('/home/yehiahesham/Desktop/ML/Ass3_data/test_images/','*.JPEG')
    files = glob.glob(path)
    X_test = []
    X_test_id = []
    total = 0
    thr = math.floor(len(files)/10)
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im(fl)
        X_test.append(img)
        X_test_id.append(flbase)
        total += 1
        if total%thr == 0:
            print('Read {} images from {}'.format(total, len(files)))

    return X_test, X_test_id


def preprocess_singleImage(x):
    x/=255.
    x-=0.5
    x*=2.
    return x

def Manual_Preporcessing(X_test):

    #manual (preprocessing of data )avg and normalizeation of testing data
    channel_axis=0
    col_axis=0
    row_axis=0

    data_format = K.image_data_format()
    if data_format == 'channels_first':
        channel_axis = 1
        row_axis = 2
        col_axis = 3
    if data_format == 'channels_last':
        channel_axis = 3
        row_axis = 1
        col_axis = 2

    mean = np.mean(X_test, axis=(0, row_axis, col_axis))
    std  = np.std(X_test, axis=(0, row_axis, col_axis))
    broadcast_shape = [1, 1, 1]
    broadcast_shape[channel_axis - 1] = X_test.shape[channel_axis]
    mean = np.reshape(mean, broadcast_shape)
    std  = np.reshape(std, broadcast_shape)
    f = open('Manual_Preporcessing', 'w')
    f.write('\n mean is \n')
    f.write(mean)
    f.write('\n std is \n')
    f.write(std)
    X_test -= mean
    X_test /= (std + K.epsilon())



# X_test,Y_test = load_test()
# print ('X_test is',len(X_test[0]))
#print ('Y_test is',Y_test[0])

# Manual_Preporcessing(X_test);

model = Xception(include_top=False, weights='imagenet', classes=200, pooling='avg', input_shape=(img_size, img_size, 3))

# for layer in model.layers:
#     layer.trainable = False

output_model = model.output
predictions = Dense(200, activation='softmax', name='predictions')(output_model)

new_model = Model(input=model.input, output=predictions)
new_model.summary()



# Loading best weights
new_model.load_weights("./weights/weights_3.hdf5")



sgd = SGD(lr=learning_rate, nesterov=True, momentum=0.7, decay=1e-4)
new_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

#--------------------------

datagen_test = ImageDataGenerator(
    featurewise_center=True,
    samplewise_center=False,
    featurewise_std_normalization=True,
    samplewise_std_normalization=False
    )

datagen_train = ImageDataGenerator(
    featurewise_center=True,
    samplewise_center=False,
    featurewise_std_normalization=True,
    samplewise_std_normalization=False,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

board = keras.callbacks.TensorBoard(log_dir='./logs/log_3', histogram_freq=0, write_graph=True, write_images=False)
checkpointer = ModelCheckpoint(filepath="./weights/weights_3.hdf5", verbose=1, save_best_only=True, monitor="val_loss")
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=0, verbose=1, mode='auto', cooldown=0, min_lr=1e-6)

train_gen_trainData  = datagen_train.flow_from_directory('/home/yehiahesham/Desktop/ML/Ass3_data/train/',
 target_size=(img_size, img_size),
 batch_size=batchSize,
 class_mode='categorical')


train_gen_testData = datagen_test.flow_from_directory('/home/yehiahesham/Desktop/ML/Ass3_data/validate/',
   target_size=(img_size, img_size),
   batch_size=batchSize,
   class_mode='categorical')

# uncommit this to train
# tensorboard --logdir=./logs

# history = new_model.fit_generator(train_gen_trainData, steps_per_epoch= 90000 / batchSize ,
#                             epochs=epochs,validation_data=train_gen_testData,validation_steps =10000 / batchSize ,verbose=1,
#                             callbacks=[board,checkpointer, reduce_lr] )


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


print ("=============================== Predicting =========================================")

l=[None]*200

for key,value in train_gen_trainData.class_indices.items():
    l[value]=key

f = open('Submission', 'w')
f.write('Id,Prediction\n')

print ("preprocess_input is OF")
print ("preprocess_singleImage is OF")
for i in range(0, 10000):
    img_name = 'test_%d' % (i,)
    img_name += '.JPEG'
    img_path = '/home/yehiahesham/Desktop/ML/Ass3_data/test_images/' + img_name
    img = image.load_img(img_path, target_size=(img_size, img_size))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)
    # x = preprocess_singleImage(x)
    preds = new_model.predict(x)
    f.write(img_name + ',' + l[np.argmax(preds)] + '\n')
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    # print('Predicted:', decode_predictions(preds, top=3)[0])
    # print()

f.close()

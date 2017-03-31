# from resnet50 import ResNet50
from keras.preprocessing import image
# from imagenet_utils import preprocess_input, decode_predictions
import os
import numpy as np
from scipy import ndimage
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator


#model = ResNet50(weights='imagenet')


train_datagen = ImageDataGenerator(
featurewise_center=False, # set input mean to 0 over the dataset
samplewise_center=False, # set each sample mean to 0
featurewise_std_normalization=False, # divide inputs by std of the dataset
samplewise_std_normalization=False, # divide each input by its std
zca_whitening=False, # apply ZCA whitening
rotation_range=0, # randomly rotate images in the range (degrees, 0 to 180)
width_shift_range=0.2, # randomly shift images horizontally (fraction of total width)
height_shift_range=0.2, # randomly shift images vertically (fraction of total height)
horizontal_flip=True, # randomly flip images
vertical_flip=False) # randomly flip images

# Augmantation on The Training data only !!!
#datagen.fit(X_train[:40000])


validate_datagen = ImageDataGenerator()


test_datagen = ImageDataGenerator()


train_generator = train_datagen.flow_from_directory(
        '/home/yehiahesham/Desktop/vi/ML/Ass3_data/train/',
        target_size=(299, 299), #299
        batch_size=128,
        class_mode='categorical',shuffle = True)


validation_generator = validate_datagen.flow_from_directory(
        '/home/yehiahesham/Desktop/vi/ML/Ass3_data/validate/',
        target_size=(299, 299), #299
        batch_size=128,
        class_mode='categorical', shuffle = True)

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(200, activation='softmax')(x)

# this is the model we will train
model = Model(input=base_model.input, output=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# train the model on the new data for a few epochs
#model.fit_generator(...)

model.fit_generator(
        train_generator,
        steps_per_epoch=90000,
        epochs=100,
        validation_data=validation_generator,
        validation_steps=10000, verbose  = 1);

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 172 layers and unfreeze the rest:
for layer in model.layers[:172]:
   layer.trainable = False
for layer in model.layers[172:]:
   layer.trainable = True


# tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/logs_1n', histogram_freq=0, write_graph=True, write_images=True)

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit_generator(
        train_generator,
        steps_per_epoch=90000,
        epochs=100,
        validation_data=validation_generator,
        validation_steps=10000, callbacks = [tensorboard], verbose = 1)



"""
test_generator = test_datagen.flow_from_directory(
        './test/',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary', shuffle = False)




predict = model.predict_generator(test_generator, 10000)

print (predict)
"""
"""
Id,Prediction
test_0.JPEG,n04399382
test_1.JPEG,n02808440
test_2.JPEG,n02808440
test_3.JPEG,n04070727
test_4.JPEG,n04067472
test_5.JPEG,n03444034
etc.
"""
"""

score = model.evaluate(X_test, Y_test, verbose=1)
print('Test score:', score[0])
print('ACCR:', score[1])
"""




"""

model.fit_generator(datagen.flow(X_train[:40000], Y_train[:40000], batch_size=batchS),
                    steps_per_epoch=X_train.shape[0]//batchS,
                    nb_epoch=nb_epoch,
                    validation_data=(X_train[40000:], Y_train[40000:])
                    # ,validation_steps=X_test.shape[0] //batch_size
                    ,callbacks=[tensorboard,checkpointer])
"""
#img_path = 'elephant.jpg'
#img = image.load_img(img_path, target_size=(224, 224))
#x = image.img_to_array(img)
#x = np.expand_dims(x, axis=0)
#x = preprocess_input(x)

#preds = model.predict(x)
#print('Predicted:', decode_predictions(preds))
# print: [[u'n02504458', u'African_elephant']]

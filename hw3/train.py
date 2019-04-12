import sys
import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, TensorBoard
from model import model_cnn, model_resnet
from utils import LoadData, PARAM


arg = sys.argv
print(len(arg))
if len(arg) < 3:
    print("Usage: python3 <train_data_path> <output_model_path>")
    exit()

train_path = arg[1]
outfile = arg[2]

## preprocess data 
## cross validation
X_train, Y_train = LoadData(train_path)
validation_rate = 0.1
index = np.int(X_train.shape[0]* (1-validation_rate))
trainX, testX, trainY, testY = X_train[:index], X_train[index:], Y_train[:index], Y_train[index:]

## data augmentation
train_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False,
    vertical_flip=True
)
test_datagen = ImageDataGenerator()
train_datagen.fit(trainX)
test_datagen.fit(testX)

## call backs functions
checkpoints = ModelCheckpoint(outfile, verbose=2, 
                              monitor='val_acc', save_best_only=True, mode='auto')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, 
                              verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.001)


BATCH  = PARAM.batch_size
EPOCHS = PARAM.epochs

## define model
models = model_cnn()
models.compile(optimizer='adam',
loss='categorical_crossentropy',
metrics=['accuracy'])
# print(models.summary())
## train the model 500 epochs
train = models.fit_generator(
    train_datagen.flow((trainX, trainY), batch_size=BATCH),
    steps_per_epoch=trainX.shape[0] // (4*BATCH),
    epochs=EPOCHS,
    validation_data=test_datagen.flow((testX,testY)),
    validation_steps=BATCH,
    callbacks=[checkpoints,reduce_lr])

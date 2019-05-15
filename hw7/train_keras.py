import utils
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import keras
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, UpSampling2D, Activation
from keras import backend as K
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

data_folder = 'data/images'
data_dirs = os.listdir(data_folder)
params = utils.PARAMS()

def read_img(i):
    image_path = str(i+1).zfill(6)
    image_path = os.path.join(data_folder, image_path+'.jpg')
    img = utils.loadImage(image_path)
    return img

import multiprocessing as mp
pool = mp.Pool()
res = pool.map(read_img, range(40000))
X_train = np.array(res)
X_train = X_train.reshape(params.img_num, params.pixel_num, params.pixel_num, params.rgb_num) / 255

# Build the autoencoder
# Build the autoencoder
kernel_size = 3
model = Sequential()
model.add(Conv2D(32, kernel_size=kernel_size, padding='same', activation='relu', 
                 input_shape=(params.pixel_num,params.pixel_num, params.rgb_num)))

model.add(MaxPool2D((2,2), padding='same'))
model.add(Dropout(0.2))

model.add(Conv2D(32, kernel_size=kernel_size, padding='same', activation='relu'))
model.add(MaxPool2D((2,2), padding='same'))
model.add(Dropout(0.2))

model.add(Conv2D(32, kernel_size=kernel_size, padding='same', activation='relu'))
model.add(UpSampling2D((2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(32, kernel_size=kernel_size, padding='same', activation='relu'))
model.add(UpSampling2D((2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(3, kernel_size=1, padding='same', activation='relu'))

model.compile(optimizer='adam', loss="mse")
model.summary()

checkpoints = ModelCheckpoint('model2.h5', save_best_only=True, monitor='val_loss', verbose=0)

from keras.preprocessing.image import ImageDataGenerator
index = X_train.shape[0] // 10 * 9
trainX, testX = X_train[:index], X_train[index:]

## data augmentation
train_datagen = ImageDataGenerator(
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
)
test_datagen = ImageDataGenerator(
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
)
train_datagen.fit(trainX)
test_datagen.fit(testX)

BATCH = 256
EPOCHS = 200
train = model.fit_generator(
    train_datagen.flow((trainX, trainX), batch_size=BATCH),
    steps_per_epoch=trainX.shape[0] // (4*BATCH),
    epochs=EPOCHS,
    validation_data=test_datagen.flow((testX,testX)),
    validation_steps=BATCH,
    callbacks=[checkpoints]
)

print(checkpoints.best)
model = load_model('model2.h5')

encoder = K.function([model.layers[0].input], [model.layers[4].output])
encoded_imgs = encoder([X_train])[0]
print(encoded_imgs.shape)
encoded_images = encoded_imgs.reshape(-1,encoded_imgs.shape[1]*encoded_imgs.shape[2]*encoded_imgs.shape[3])


# Cluster the training set
kmeans = KMeans(n_clusters=11, n_jobs=-1)
clustered_training_set = kmeans.fit_predict(encoded_images)

from sklearn.metrics.pairwise import cosine_similarity
sim=cosine_similarity(encoded_images, dense_output=True)

test=pd.read_csv('data/test_case.csv').values
ans = []
for cnt,i in enumerate(test):
    class1, class2 = i[1] - 1, i[2]-1
    class1, class2 = clustered_training_set[class1] , clustered_training_set[class2]
    ans.append([cnt,int(class1==class2)])

threshold = np.mean(sim)
print(threshold)

test = pd.read_csv('data/test_case.csv').values
ans = []
for cnt,i in enumerate(test):
    class1, class2 = i[1]-1, i[2]-1
    sim_score = 1 if sim[class1, class2] > threshold else 0
    ans.append([cnt,sim_score])

df=pd.DataFrame(ans,columns=['id', 'label'])
# df.to_csv('data/ans5.csv', index=None)

begin = 0
print(clustered_training_set[begin:begin+10])
X_test=model.predict(X_train)[begin:begin+10]
# Observe the reconstructed image quality
plt.figure(figsize=(20,5))
for i in range(10):
    plt.subplot(2, 10, i+1)
    plt.imshow(X_test[i])
    
    plt.subplot(2, 10, i+11)
    plt.imshow(X_train[i+begin])
plt.savefig('sample.png')


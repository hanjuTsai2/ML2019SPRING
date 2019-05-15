#!/usr/bin/env python
# coding: utf-8

# In[1]:


import utils
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import keras
from keras.models import Sequential, load_model, Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, UpSampling2D, Activation, BatchNormalization, Input, MaxPooling2D
from keras.layers.advanced_activations import PReLU
from keras import backend as K
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import keras.backend as K


# In[2]:


data_folder = 'data/images'
data_dirs = os.listdir(data_folder)
params = utils.PARAMS()


# In[3]:


def read_img(i):
    image_path = str(i+1).zfill(6)
    image_path = os.path.join(data_folder, image_path+'.jpg')
    img = utils.loadImage(image_path)
    return img


# In[4]:


import multiprocessing as mp
pool = mp.Pool()
res = pool.map(read_img, range(40000))
pool.close()


# In[5]:


X_train = np.array(res)


# In[6]:


X_train = X_train.reshape(params.img_num, params.pixel_num, params.pixel_num, params.rgb_num) / 255


# In[33]:


from keras.preprocessing.image import ImageDataGenerator
index = X_train.shape[0] // 10 * 9
trainX, testX = X_train[:index], X_train[index:]

## data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
)
test_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
)
train_datagen.fit(trainX)
test_datagen.fit(testX)


# In[66]:


kernel_size = 3
model = Sequential()
model.add(Conv2D(32, kernel_size=kernel_size, padding='same', activation='relu', 
                 input_shape=(params.pixel_num,params.pixel_num, params.rgb_num)))

model.add(MaxPool2D((2,2), padding='same'))
# model.add(Dropout(0.1))

model.add(Conv2D(16, kernel_size=kernel_size, padding='same', activation='relu'))
model.add(MaxPool2D((2,2), padding='same'))
# model.add(Dropout(0.1))

model.add(Conv2D(16, kernel_size=kernel_size, padding='same', activation='relu'))
model.add(UpSampling2D((2,2)))


model.add(Conv2D(32, kernel_size=kernel_size, padding='same', activation='relu'))
model.add(UpSampling2D((2,2)))
# model.add(Dropout(0.2))

model.add(Conv2D(3, kernel_size=1, padding='same', activation='relu'))

model.compile(optimizer='adam', loss="mse")
model.summary()


# In[67]:


checkpoints = ModelCheckpoint('model6.h5', save_best_only=True, monitor='val_loss', verbose=1)


# In[68]:


noise_factor = 0.5
trainX_noisy = trainX + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=trainX.shape)
testX_noisy = testX + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=testX.shape)


# In[74]:


model.fit(trainX, trainX, validation_data=(testX,testX),
                batch_size=1024, epochs=200, callbacks=[checkpoints])


# In[ ]:


model.fit(trainX_noisy, trainX, validation_data=(testX_noisy,testX),
                batch_size=512, epochs=400, callbacks=[checkpoints])


# In[ ]:


BATCH = 512
EPOCHS = 200
train = model.fit_generator(
    train_datagen.flow((trainX, trainX), batch_size=BATCH),
    steps_per_epoch=trainX.shape[0] // (4*BATCH),
    epochs=EPOCHS,
    validation_data=test_datagen.flow((testX,testX)),
    validation_steps=BATCH,
    callbacks=[checkpoints])


# In[7]:


K.set_learning_phase(0)
model = load_model('model3.h5')
model.summary()


# In[56]:


# model.save('model6.h5')


# In[76]:


model = load_model('model6.h5')


# In[77]:


# model = autoencoder
encoder = K.function([model.layers[0].input], [model.layers[3].output])


# In[78]:


encoded_imgs = encoder([X_train])[0]
print(encoded_imgs.shape)


# In[79]:


encoded_images = encoded_imgs.reshape(-1,encoded_imgs.shape[1]*encoded_imgs.shape[2]*encoded_imgs.shape[3])


# In[21]:


# from sklearn.decomposition import KernelPCA
# kpca = KernelPCA(n_components=100, kernel="rbf", fit_inverse_transform=True, gamma=10, copy_X=False)
# image_X_PCA  = kpca.fit_transform(encoded_images)


# In[80]:


pca = PCA(n_components=400, whiten=True, svd_solver="full", random_state=0)
image_X_PCA = pca.fit_transform(encoded_images)


# In[21]:


# from sklearn.manifold import TSNE
# image_X_PCA = TSNE(n_components=2, verbose=1, n_iter=250).fit_transform(encoded_images)


# In[81]:


print("Run KMeans")
kmeans = KMeans(n_clusters=2, random_state=0)
clustered_training_set = kmeans.fit_predict(image_X_PCA)


# In[ ]:


from sklearn import cluster
spectral = cluster.SpectralClustering(n_clusters=2,
                                      eigen_solver='arpack',
                                      affinity="nearest_neighbors")
clustered_training_set = spectral.fit_predict(image_X_PCA)


# In[ ]:


# from sklearn.cluster import DBSCAN

# clustering = DBSCAN(eps=2, min_samples=2, n_jobs=-1).fit(encoded_images)
# clustering.labels_


# In[82]:


test = pd.read_csv('data/test_case.csv').values
ans = []
for cnt, i in enumerate(test):
    class1, class2 = i[1] - 1, i[2] - 1
    class1, class2 = clustered_training_set[class1] , clustered_training_set[class2]
    ans.append([cnt,int(class1==class2)])

df = pd.DataFrame(ans,columns=['id', 'label'])


# In[83]:


df_ans = pd.read_csv('pca10.csv')
print(np.mean(df.label == df_ans.label))


# In[85]:


np.sum(df.label==1), np.sum(df.label==0)


# In[86]:


df.to_csv('pca11.csv', index=None)


# In[144]:


df = pd.DataFrame(ans,columns=['id', 'label'])
# df.to_csv('data/ans15.csv', index=None)


# In[75]:


ranges = range(10)
clu = (clustered_training_set[ranges])
X_test = model.predict(X_train)[ranges]
# Observe the reconstructed image quality
plt.figure(figsize=(20,5))
for i in range(10):
    plt.subplot(2, 10, i+1)
    plt.imshow(X_test[i])
    plt.title(clu[i])
    
    plt.subplot(2, 10, i+11)
    plt.imshow(X_train[i+ranges[0]])
# plt.savefig('sample.png')


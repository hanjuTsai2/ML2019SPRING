#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
import keras.regularizers as regularizers
from keras import backend as K
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA


# In[3]:


data_folder = 'data/images'
data_dirs = os.listdir(data_folder)
params = utils.PARAMS()


# In[4]:


def read_img(i):
    image_path = str(i+1).zfill(6)
    image_path = os.path.join(data_folder, image_path+'.jpg')
    img = utils.loadImage(image_path)
    return img


# In[5]:


import multiprocessing as mp
pool = mp.Pool()
res = pool.map(read_img, range(40000))
pool.close()


# In[24]:


X_train = np.array(res)/255


# In[25]:


X_flatten=X_train.reshape(-1, 32*32*3)


# In[7]:


# X_train = X_train.reshape(params.img_num, params.pixel_num, params.pixel_num, params.rgb_num)# / 255


# In[105]:


input_dim = X_flatten.shape[1]
encoding_dim = 2  
input_img = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='sigmoid')(input_img)
decoded = Dense(input_dim, activation='sigmoid')(encoded)
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
print(autoencoder.summary())


# In[100]:


encoding_dim = 400
# this is our input placeholder
input_img = Input(shape=(32*32*3,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(32*32*3, activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')


# In[101]:


input_dim = X_train.shape[1]
encoding_dim = 2
input_img = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='linear', activity_regularizer=regularizers.l2(10e-5))(input_img)
decoded = Dense(input_dim, activation='linear')(encoded)
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
print(autoencoder.summary())


# In[86]:


m = Sequential()
init_dens = 256
m.add(Dense(init_dens,  activation='elu', input_shape=(32*32*3,)))
m.add(Dense(init_dens//2,  activation='elu'))
m.add(Dense(init_dens//4,    activation='linear', name="bottleneck"))
m.add(Dense(init_dens//2,  activation='elu'))
m.add(Dense(init_dens,  activation='elu'))
m.add(Dense(32*32*3,  activation='sigmoid'))
m.compile(optimizer='adam', loss='mse')
m.summary()


# In[106]:


history = autoencoder.fit(X_flatten, X_flatten, batch_size=128, epochs=100, verbose=1, validation_split=0.1)


# In[88]:


autoencoder.fit(X_train, X_train, validation_split=.1,
                batch_size=512, epochs=400)#, callbacks=[checkpoints])


# In[43]:


model = m


# In[44]:


encoder = K.function([model.layers[0].input], [model.layers[2].output])


# In[45]:


encoded_imgs = encoder([X_flatten])[0]
encoded_imgs.shape


# In[9]:


pca = PCA(n_components=400, whiten=True,svd_solver="full",random_state=0)
image_X_PCA = pca.fit_transform(X_flatten)

print("Run KMeans")
cluster = KMeans(n_clusters=2)
cluster.fit(image_X_PCA)


# In[26]:


X_pca = pca.inverse_transform(image_X_PCA).reshape(-1,32,32,3)
X_pca = np.clip(X_pca,0,1)


# In[48]:


test = pd.read_csv('data/test_case.csv').values
ans = []
for cnt, i in enumerate(test):
    class1, class2 = i[1] - 1, i[2] - 1
    class1, class2 = cluster.labels_[class1] , cluster.labels_[class2]
    ans.append([cnt,int(class1==class2)])


# In[49]:


df = pd.DataFrame(ans,columns=['id', 'label'])


# In[50]:


ans = pd.read_csv('data/pca.csv')


# In[51]:


np.mean(ans.label == df.label)


# In[66]:


df = pd.DataFrame(ans,columns=['id', 'label'])
# df.to_csv('data/pca2.csv', index=None)


# In[132]:


X_test = autoencoder.predict(X_train)[ranges]


# In[30]:


ranges = range(10)
# X_test = autoencoder.predict(X_flatten)[ranges]
# X_test = X_test.reshape(-1,32,32,3)
# Observe the reconstructed image quality
plt.figure(figsize=(20,5))
for i in range(10):
    plt.subplot(2, 10, i+1)
#     plt.imshow(X_test[i])
    plt.imshow(X_pca[i+ranges[0]])
    
    plt.subplot(2, 10, i+11)
    plt.imshow(X_train[i+ranges[0]])
plt.savefig('sample.png')


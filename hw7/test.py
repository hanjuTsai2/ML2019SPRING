import sys
import utils
import numpy as np
import pandas as pd
import os
import datetime
import multiprocessing as mp

import keras
from keras.models import Sequential, load_model, Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, UpSampling2D, Activation, BatchNormalization, Input, MaxPooling2D
from keras.layers.advanced_activations import PReLU
from keras import backend as K

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

def read_img(i):
    image_path = str(i+1).zfill(6)
    image_path = os.path.join(data_folder, image_path +'.jpg')
    img = utils.loadImage(image_path)
    return img

def cluster_GAUSSIAN(i, iterations, encoded_images):
    if iterations != 0:
        pca = PCA(n_components=i, whiten=True, svd_solver = 'full', random_state=0)
        image_X_PCA = pca.fit_transform(encoded_images)
    else:
        image_X_PCA = encoded_images
    clustered_training_set=GaussianMixture(2, covariance_type='tied',
                                           max_iter=iterations, tol=1e-5,
                                           random_state=0, warm_start=True).fit_predict(image_X_PCA)
    
        
    return clustered_training_set, image_X_PCA


def cluster_PCA(i, iterations, encoded_images):
    pca = PCA(n_components=i, whiten=True, svd_solver = 'full', random_state=0)
    image_X_PCA = pca.fit_transform(encoded_images)
    kmeans = KMeans(n_clusters=2, random_state=0, precompute_distances=True)
    clustered_training_set = kmeans.fit_predict(image_X_PCA)
        
    return clustered_training_set, image_X_PCA

def main(argv):
    start = datetime.datetime.now()
    print(argv)
    
    assert(len(argv) >= 4)
    global data_folder
    data_folder = argv[1]
    test_data = argv[2]
    output = argv[3]
    
    data_dirs = os.listdir(data_folder)
    params = utils.PARAMS()

    pool = mp.Pool()
    res = pool.map(read_img, range(40000))
    pool.close()
    X_train = np.array(res)
    X_train = X_train.reshape(params.img_num,
        params.pixel_num,
        params.pixel_num,
        params.rgb_num) / 255 

    model = load_model('model/model28.h5')
    encoder = Model([model.get_layer('input_layer').input], [model.get_layer('code_layer').output])
    encoded_imgs = encoder.predict(X_train,batch_size=128)
    print(encoded_imgs.shape)
    encoded_images = encoded_imgs.reshape(-1,encoded_imgs.shape[1]*encoded_imgs.shape[2]*encoded_imgs.shape[3])

    for i in range(1100, encoded_images.shape[1], 100): 
        clustered_training_set, image_X_PCA = cluster_PCA(i, 4000, encoded_images)
        break

    test = pd.read_csv(test_data).values
    ans = []
    for cnt, i in enumerate(test):
        class1, class2 = i[1] - 1, i[2] - 1
        class1, class2 = clustered_training_set[class1] , clustered_training_set[class2]
        ans.append([cnt,int(class1==class2)])

    df = pd.DataFrame(ans,columns=['id', 'label'])
    df.to_csv(output, index=None)
    end = datetime.datetime.now()
    print("{} mins".format((end-start).seconds//60))

if __name__ == '__main__':
    main(sys.argv)
import sys
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.utils import to_categorical, plot_model
from keras.models import load_model
from keras import backend as K
from numpy import random

import lime
from lime import lime_image
from skimage.color import grey2rgb, rgb2gray
from skimage.segmentation import mark_boundaries
from lime.wrappers.scikit_image import SegmentationAlgorithm

seed = 5
random.seed(seed)

def LoadData(path):
    train = pd.read_csv(path)
    Y_train = np.array(train['label'])
    Y_train_cat = to_categorical(Y_train)
    
    X = np.array(train['feature'])
    X_train = []
    for i in range(X.shape[0]):
        x = np.array(X[i].split(' '))
        x = x.astype(np.int)/255
        X_train.append(x)
    X_train = np.array(X_train)
    X_train = X_train.reshape(X_train.shape[0],48,48,1) 
    return X_train, Y_train
    
def plotSaliencyMap(model, X_train, Y_train, image_list, lables_dict, folder):  
    ## selected images
    for cnt, i in enumerate(image_list):
        #Y_prob = model.predict(X_train[i].reshape(-1,48,48,1))
        Y_pred = [cnt] #Y_prob.argmax(axis=-1)

        inputs = model.input
        outputs = model.output
        tensorGradients = K.gradients(outputs, inputs)[0]
        funct = K.function([inputs], [tensorGradients])

        ### start heatmap processing ###
        gradients = funct([X_train[i].reshape(-1, 48, 48, 1), 0])[0].reshape(48, 48, -1)
        gradients = np.max(np.abs(gradients), axis=-1, keepdims=True)
        gradients = (gradients - np.mean(gradients)) / (np.std(gradients) + 10 ** -5) + 0.5

        # clip to [0, 1]
        gradients = np.clip(gradients, 0, 1)
        heatmap = gradients.reshape(48, 48)
        
        # original
        fig = plt.figure(figsize=(10, 3))
        ax = fig.add_subplot(1, 3, 1)
        axx = ax.imshow((X_train[i] * 255).reshape(48, 48), cmap="gray")
        plt.tight_layout()
    
        # heat map
        ax = fig.add_subplot(1, 3, 2)
        ax.set_title(str(lables_dict[cnt]) + ' id ' + str(i))
        axx = ax.imshow(heatmap, cmap=plt.cm.jet)
        plt.colorbar(axx)
        plt.tight_layout()

        # salency map
        showedpixel = (X_train[i] * 255).reshape(48, 48)
        showedpixel[np.where(heatmap < np.mean(heatmap))] = np.mean(showedpixel)
    
        ax = fig.add_subplot(1, 3, 3)
        axx = ax.imshow(showedpixel, cmap="gray")
        
        plt.colorbar(axx)
        plt.tight_layout()
        plt.savefig(folder+"fig1_" + str(cnt))


## reference from https://keras.io/examples/conv_filter_visualization/
## normalize tensor function
def normalize(x):
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.25
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.25

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def trainGradAscent(steps, images, functions, record_freq):
    result = []
    lr = 1e-2
    for i in range(steps):
        loss, gradients = functions([images, 0])
        images += gradients * lr
        if i % record_freq == 0:
            result.append((images, loss))
    return result

def plotFilters(model, folder, draw_first_only=True):
    steps_num = 40
    steps = 40
    record_freq = 20
    filter_num = 64
    intFilters = 64

    dictLayer = dict([layer.name, layer] for layer in model.layers)
    inputImage = model.input
    listLayerNames = [layer for layer in dictLayer.keys() if "activation" in layer or "conv2d" in layer][:8]
    if draw_first_only:
        listLayerNames = [ listLayerNames[1] ]
    layers = [dictLayer[name].output for name in listLayerNames]

    for cnt, fn in enumerate(layers):
        listFilterImages = []
        for i in range(intFilters):
            images = np.random.random((1, 48, 48, 1)) 
            tensorTarget = K.mean(fn[:, :, :, i])

            gradients = normalize(K.gradients(tensorTarget, inputImage)[0])
            functions = K.function([inputImage, K.learning_phase()], [tensorTarget, gradients])
            listFilterImages.append(trainGradAscent(steps, images, functions, record_freq))
    
        times = steps_num // record_freq
        for it in range(times):
            fig = plt.figure(figsize=(16, 17))
            for i in range(filter_num):
                ax = fig.add_subplot(filter_num/8, 8, i+1)
                raw = listFilterImages[i][it][0].squeeze()
                ax.imshow(deprocess_image(raw), cmap="Reds")
                plt.xticks(np.array([]))
                plt.yticks(np.array([]))
                plt.tight_layout()
            plt.savefig(folder+"fig2_1")

def plotImageFiltersResult(model, X_train, idx, folder , draw_first_only = True):
    image_w,image_h = 48, 48

    dictLayer = dict([layer.name, layer] for layer in model.layers)
    inputImage = model.input
    
    layers_name = [layer for layer in dictLayer.keys() if "activation" in layer or "conv2d" in layer][:8]
    if draw_first_only:
        layers_name = [layers_name[1]]
    layers = [K.function([inputImage, K.learning_phase()], [dictLayer[name].output]) for name in layers_name]
    
    for cnt, fn in enumerate(layers):
        images = X_train[idx].reshape(1,image_w,image_h, 1)
        images = fn([images, 0]) 
        fig = plt.figure(figsize=(16, 16))
        filter_num = 64
        for i in range(filter_num):
            ax = fig.add_subplot(8, 8, i+1)
            ax.imshow(images[0][0, :, :, i], cmap="Reds")
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.xlabel("filter {}".format(i))
            plt.tight_layout()
        fig.suptitle("{}-given{} th images".format(layers_name[cnt], idx))
        plt.savefig(folder+"fig2_2")


def prediction_function(x):
    x = x[:,:,:,0:1]/255
    result =(model.predict(x))
    return result

def plotLime(folder, image_list):
    global seed
    segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2, random_seed=seed)
    explainer = lime_image.LimeImageExplainer(random_state=seed)
    for cnt, idx in enumerate(image_list):
        sample = x_train_rgb[idx]
        explanation = explainer.explain_instance(sample, 
                                             classifier_fn = prediction_function, 
                                             top_labels=7,
                                             hide_color=1,
                                             num_samples=100,
                                             random_seed=seed,
                                             segmentation_fn=segmenter)
        temp, mask = explanation.get_image_and_mask(cnt, positive_only=True, num_features=3, hide_rest=False)
        plt.imshow(mark_boundaries(temp, mask))
        plt.savefig(folder+ "fig3_" + str(cnt))



arg = sys.argv
path = arg[1]
folder = arg[2]

lables = ["angry", "disgust" , "fear" , "happy" , "sad", "surprise", "neutral"]
lables_dict = {0: "angry", 1:"disgust", 2:"fear", 3: "happy", 4:"sad", 5:"surprise", 6:"neutral"}
image_list = [1, 14779, 7128, 3742, 69, 199, 16529]

X_train, Y_train = LoadData(path)
x_train_rgb = grey2rgb(X_train * 255)
x_train_rgb = x_train_rgb.astype(np.uint8).reshape(X_train.shape[0],X_train.shape[1], X_train.shape[1], 3)

model_file = 'hw4_model/model31.h5'
model = load_model(model_file)

plotLime(folder, image_list)
plotFilters(model, folder)
plotImageFiltersResult(model,X_train,2, folder)
plotSaliencyMap(model, X_train , Y_train, image_list, lables_dict, folder)


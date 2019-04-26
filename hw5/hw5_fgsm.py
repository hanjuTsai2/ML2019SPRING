import keras
import pandas as pd
import numpy as np
from PIL import Image
import keras.backend as K
from copy import deepcopy
from keras.applications import resnet50, vgg16, vgg19, densenet
from keras_applications.resnet import ResNet101
from keras.applications.vgg16 import VGG16
K.set_learning_phase(0)

## set up path
import sys
import os
input_dir = sys.argv[1]
output_dir = sys.argv[2]

class PARAMS:
    img_num = 200
    pixel_num = 224
    rgb_num = 3
    
def array2PIL(np_im, path):
    np_im = np_im.reshape(224,224,3).astype(dtype=np.uint8)
    new_im = Image.fromarray(np_im)
    new_im.save(path)
      
def plot_img(x, path=None):
    x = np.clip((x), 0, 255)
    if path:
        array2PIL(x, path)
        
def array2Images(xss, folder, ranges=range(PARAMS.img_num)):
    for cnt, i in enumerate(ranges):
        img_path = os.path.join(folder, '{:03d}'.format(i) + ".png")
        xs = xss[cnt].reshape(1, PARAMS.pixel_num, PARAMS.pixel_num, PARAMS.rgb_num)
        plot_img(xs, path=img_path)

def loadImage(img_path):
    img = Image.open(img_path).convert('RGB')
    img = np.array(img)
    return img

def loadFolderImages(folder, ranges=None):
    if not ranges:
        ranges = range(PARAMS.img_num)
        
    images= np.array([])
    for i in (ranges):
        img_path = os.path.join(folder, '{:03d}'.format(i) + ".png")
        img = loadImage(img_path)
        images = np.append(images, img)
    return images

def l1InfNormFolder(input_folder, output_folder):
    inputs = loadFolderImages(input_folder)
    outputs = loadFolderImages(output_folder)  
    return l1InfNorm(inputs, outputs)

def l1InfNorm(inputs, outputs):
    inputs = inputs.ravel()
    outputs = outputs.ravel()
    inputs = inputs.astype(dtype=np.uint8)
    outputs = outputs.astype(dtype=np.uint8)
    diff = np.abs(inputs - outputs)
    diff = diff.reshape(PARAMS.img_num, PARAMS.pixel_num * PARAMS.pixel_num * PARAMS.rgb_num)
    max_diff = np.max(diff, axis=1)
    norm = np.mean(max_diff)
    return norm

def postprocess_input_v1(prep):
    return (prep + [103.939, 116.779, 123.68])[..., ::-1]

def postprocess_input_v2(prep):
    return (prep * [0.229, 0.224, 0.225] +  [0.485, 0.456, 0.406])*255


batch_size = 20
iteration = int(PARAMS.img_num / batch_size)
X_train = np.array([])
X_test = np.array([])
model = VGG16()

for j in range(iteration):
    start, end = j*batch_size, np.clip((j+1) * batch_size, 0, PARAMS.img_num)
    images = loadFolderImages(input_dir , range(start, end))
    x = images.reshape(batch_size, PARAMS.pixel_num, PARAMS.pixel_num, PARAMS.rgb_num)
    
    preds = model.predict(x)
    initial_class = np.argmax(preds,axis=1)
    initial_index = vgg16.decode_predictions(preds, top=3)
    initial_index = np.array([i[0][0] for i in initial_index])

    # Get current session (assuming tf backend)
    sess = K.get_session()
    # Initialize adversarial example with input image
    x_adv = x
    # Added noise
    x_noise = np.zeros_like(x)

    # Set variables
    epsilon = 5
    prev_probs = []

    # One hot encode the initial class
    target = K.one_hot(initial_class, 1000)

    # Get the loss and gradient of the loss wrt the inputs
    loss = K.categorical_crossentropy(target, model.output)
    grads = K.gradients(loss, model.input)

    # Get the sign of the gradient
    delta = K.sign(grads[0])
    x_noise = x_noise + delta

    # Perturb the image
    x_adv = x_adv + epsilon * delta

    # Get the new image and predictions
    x_adv = sess.run(x_adv, feed_dict={model.input:x})
    preds = model.predict(x_adv)

    first_index = vgg16.decode_predictions(preds, top=3)
    first_index = np.array([ i[0][0] for i in first_index])

    acc = np.mean(first_index != initial_index)

    ## attack successful  
    print(acc)
    array2Images(x_adv, output_dir , ranges=range(start, end))

import keras
import pandas as pd
import numpy as np
from PIL import Image
import keras.backend as K
import matplotlib.pyplot as plt
from copy import deepcopy
from keras.applications import resnet50, vgg16, vgg19, densenet
from keras_applications.resnet import ResNet101
K.set_learning_phase(0)

## set up path
import sys
import os
begin_iter = sys.argv[1]
input_dir = sys.argv[2]
output_dir = sys.argv[3]

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

def _preprocess_numpy_input(x, data_format, mode):
    """Preprocesses a Numpy array encoding a batch of images.
    # Arguments
        x: Input array, 3D or 4D.
        data_format: Data format of the image array.
        mode: One of "caffe", "tf" or "torch".
            - caffe: will convert the images from RGB to BGR,
                then will zero-center each color channel with
                respect to the ImageNet dataset,
                without scaling.
            - tf: will scale pixels between -1 and 1,
                sample-wise.
            - torch: will scale pixels between 0 and 1 and then
                will normalize each channel with respect to the
                ImageNet dataset.
    # Returns
        Preprocessed Numpy array.
    """
    x = x.astype(np.float, copy=False)

    if mode == 'tf':
        x /= 127.5
        x -= 1.
        return x

    if mode == 'torch':
        x /= 255.
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        if data_format == 'channels_first':
            # 'RGB'->'BGR'
            if x.ndim == 3:
                x = x[::-1, ...]
            else:
                x = x[:, ::-1, ...]
        else:
            # 'RGB'->'BGR'
            x = x[..., ::-1]
        mean = [103.939, 116.779, 123.68]
        std = None

    # Zero-center by mean pixel
    if data_format == 'channels_first':
        if x.ndim == 3:
            x[0, :, :] -= mean[0]
            x[1, :, :] -= mean[1]
            x[2, :, :] -= mean[2]
            if std is not None:
                x[0, :, :] /= std[0]
                x[1, :, :] /= std[1]
                x[2, :, :] /= std[2]
        else:
            x[:, 0, :, :] -= mean[0]
            x[:, 1, :, :] -= mean[1]
            x[:, 2, :, :] -= mean[2]
            if std is not None:
                x[:, 0, :, :] /= std[0]
                x[:, 1, :, :] /= std[1]
                x[:, 2, :, :] /= std[2]
    else:
        x[..., 0] -= mean[0]
        x[..., 1] -= mean[1]
        x[..., 2] -= mean[2]
        if std is not None:
            x[..., 0] /= std[0]
            x[..., 1] /= std[1]
            x[..., 2] /= std[2]
    return x

def postprocess_input_v1(prep):
    return (prep + [103.939, 116.779, 123.68])[..., ::-1]

def postprocess_input_v2(prep):
    return (prep * [0.229, 0.224, 0.225] +  [0.485, 0.456, 0.406])*255

# set models
models = [
    #("ResNet50", resnet50.ResNet50, resnet50.preprocess_input,postprocess_input_v1, 5),
#     ("ResNet101", lambda : ResNet101(backend=keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils), resnet50.preprocess_input, postprocess_input_v1, 5),
    ("VGG16", vgg16.VGG16, vgg16.preprocess_input,postprocess_input_v1, 5),
#     ("VGG19", vgg19.VGG19, vgg19.preprocess_input,postprocess_input_v1, 10 ),
#     ("DenseNet121", densenet.DenseNet121, densenet.preprocess_input,postprocess_input_v2, 5/255/0.229),
    #("DenseNet169", densenet.DenseNet169, densenet.preprocess_input,postprocess_input_v2, 5/255/0.229)
] 

# Set variables
batch_size = 20
iteration = int(PARAMS.img_num / batch_size)

assert(PARAMS.img_num % batch_size == 0)
for outdir, struct, pre_input, post_input, total_eps in models:
    model = struct()
    
    epochs = 1 * 10
    ## allocate two process
    epsilon = total_eps / epochs / 2
    
    for j in range(int(begin_iter),int(begin_iter)+1):
        print('begin iteration', begin_iter)
        start, end = j*batch_size, np.clip((j+1) * batch_size, 0, PARAMS.img_num)
        print(start,end)
        images = loadFolderImages(input_dir, range(start, end))
        x_ori = images.reshape(batch_size, PARAMS.pixel_num, PARAMS.pixel_num, PARAMS.rgb_num)
        x = pre_input(deepcopy(x_ori))

        preds = model.predict(x)
        initial_class = np.argmax(preds,axis=1)
        initial_index = vgg16.decode_predictions(preds, top=3)
        initial_index = np.array([i[0][0] for i in initial_index])

#         preds -= preds.max(axis=0)
#         preds[np.where(preds > -0.3)] = 0
#         preds *= -1
#         least_class = np.argsort(preds)[:,-1]
#         least_index = vgg16.decode_predictions(preds, top=3)
#         least_index = np.array([i[0][0] for i in least_index])
        least_class = np.argsort(preds)[:,-2]
        least_index = vgg16.decode_predictions(preds, top=3)
        least_index = np.array([i[1][0] for i in least_index])
        print('initial', np.mean([preds[cnt,i] for cnt, i in enumerate(initial_class)]))
        
        # Get current session (assuming tf backend)
        sess = K.get_session()

        # Initialize adversarial example with input image
        x_adv = x

        for i in range(epochs):
            # One hot encode the initial class
            target = K.one_hot(initial_class, 1000)

            # Get the loss and gradient of the loss wrt the inputs
            loss = K.categorical_crossentropy(target, model.output)
            grads = K.gradients(loss, model.input)

            # Get the sign of the gradient
            delta = K.sign(grads[0])
            
            # Perturb the image
            x_adv = x_adv + epsilon * delta

            # Get the new image and predictions
            x_adv = sess.run(x_adv, feed_dict={model.input:x})
            preds = model.predict(x_adv)

            first_index = vgg16.decode_predictions(preds, top=3)
            first_index = np.array([ i[0][0] for i in first_index])
            print(outdir , j, 'times' , i, 'epochs')
            acc = np.mean(first_index != initial_index)
            print('acc:', acc)
            print('top 1' , np.mean(preds.max(axis=1)))
            print('initial', np.mean([preds[cnt,i] for cnt, i in enumerate(initial_class)]))
            if acc == 1:
                del delta, loss, target, acc, first_index
                break
                
            del delta, loss, target, acc, first_index
        del preds, grads
        
        preds = model.predict(x_adv)
        all_seq = np.argsort(preds)
        
        least_index = vgg16.decode_predictions(preds, top=3)
        least_class = np.array([ all_seq[cnt][-1] if i[0][0] != initial_index[cnt] else all_seq[cnt][-2] 
                                for cnt, i in enumerate(least_index)])
        least_index = np.array([ i[0][0] if i[0][0] != initial_index[cnt] else i[1][0] 
                                for cnt, i in enumerate(least_index)])
        
        t_acc = 0.0
        epochs += epochs - i
        for i in range(epochs):
            print('begin iteration',begin_iter)
            # One hot encode the initial class
            target = K.one_hot(least_class, 1000)

            # Get the loss and gradient of the loss wrt the inputs
            loss = -1 * K.categorical_crossentropy(target, model.output)
            grads = K.gradients(loss, model.input)

            # Get the sign of the gradient
            delta = K.sign(grads[0])
            # Perturb the image
            x_adv = x_adv + epsilon * delta

            gradient = grads
            gradient_norm = K.sqrt(K.mean(K.square(grads[0])))
            x_adv = x_adv + grads[0] / (gradient_norm+1e-8) / 255 * epsilon

#           Get the new image and predictions
            x_adv = sess.run(x_adv, feed_dict={model.input:x})
            preds = model.predict(x_adv)

            first_index = vgg16.decode_predictions(preds, top=3)
            first_index = np.array([ i[0][0] for i in first_index])
            print(outdir , j, 'times' , i, 'epochs')

            ## predict vs. initial labels
            acc = np.mean(first_index != initial_index)

            now_t_acc = np.mean(first_index == least_index)
            
            if now_t_acc < t_acc:
                x_adv = x_adv - epsilon * delta
                x_adv = sess.run(x_adv, feed_dict={model.input:x})
                break
                
            # predict vs. true labels
            t_acc = now_t_acc

            # attack successful  
            print('top 1' , np.mean(preds.max(axis=1)))
            print('initial', np.mean([preds[cnt,i] for cnt, i in enumerate(initial_class)]))
            print('acc:', acc, end = ', ')
            print('target_acc:',t_acc)
            del delta, loss, target, acc, first_index, preds, grads
        
        epochs = epochs - i
        print(epochs)
        for i in range(epochs):
            # One hot encode the initial class
            target = K.one_hot(initial_class, 1000)

            # Get the loss and gradient of the loss wrt the inputs
            loss = K.categorical_crossentropy(target, model.output)
            grads = K.gradients(loss, model.input)

            # Get the sign of the gradient
            delta = K.sign(grads[0])
            
            # Perturb the image
            x_adv = x_adv + epsilon * delta

            # Get the new image and predictions
            x_adv = sess.run(x_adv, feed_dict={model.input:x})
            preds = model.predict(x_adv)

            first_index = vgg16.decode_predictions(preds, top=3)
            first_index = np.array([ i[0][0] for i in first_index])
            print(outdir , j, 'times' , i, 'epochs')
            acc = np.mean(first_index != initial_index)
            print('acc:', acc)
            print('top 1' , np.mean(preds.max(axis=1)))
            print('initial', np.mean([preds[cnt,i] for cnt, i in enumerate(initial_class)]))
            
        x_adv = np.round(post_input(x_adv))
        print('L-inf :', np.max(np.abs(x_ori.astype(np.int)-x_adv.astype(np.int))))
        array2Images(x_adv, output_dir , ranges=range(start, end))
        del images, x_adv, x_ori, least_class, least_index, sess
        
    del model, epochs, epsilon
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class PARAMS:
    img_num = 40000
    pixel_num = 32
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
        img_path =  folder + str(i+1).zfill(5) + ".png"
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
        img_path = folder +  str(i+1).zfill(5)  + ".png"
        img = loadImage(img_path)
        images = np.append(images, img)
    return images
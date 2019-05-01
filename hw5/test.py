import numpy as np
import torch 
import torch.nn as nn
import torchvision.transforms as transform
from torch.autograd.gradcheck import zero_gradients
from PIL import Image
from scipy.misc import imsave
from torch.autograd import Variable
from torchvision.models import vgg16, vgg19, resnet50, \
                               resnet101, densenet121, densenet169 

from torch.autograd import Variable
import torch.nn.functional as F 
import torchvision 
import torch.optim as optim 
from torchvision import transforms
import torchvision.models 

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
        img_path =  folder + '{:03d}'.format(i) + ".png"
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
        img_path = folder + '{:03d}'.format(i) + ".png"
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

with open("./labels.json","r") as f: 
    import json 
    ImageNet_mapping = json.loads(f.read())
    
def image_location_generator(_root):
    import os
    _dirs = os.listdir(_root)
    _dirs = [ os.path.join(_root, _dir) for _dir in _dirs]
    return _dirs

imsize = (224, 224)
loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def image_loader(image_name):
    """load image, returns tensor"""
    image = Image.open(image_name)
    image = image.convert("RGB")
    image = loader(image).float()
    image = normalize(image).float()
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image 

def defense_image_loader(image_name):
    """load image, returns tensor"""
    #3*3 Gassian filter
    x, y = np.mgrid[-1:2, -1:2]
    gaussian_kernel = np.exp(-(x**2+y**2))
    #Normalization
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum() * 1.5
    image = Image.open(image_name)
    image = image.convert("RGB")
    image = loader(image).float()
    img = image.data.numpy()
    for i in range(3):
        img[:,:,i]= signal.convolve2d(img[:,:,i], gaussian_kernel, boundary='symm', mode='same')
    image = torch.Tensor(img)
    image = normalize(image).float()
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image 

def tensor2im(tens):  
    im = tens[0]
    im[0] = im[0] * 0.229
    im[1] = im[1] * 0.224 
    im[2] = im[2] * 0.225 
    im[0] += 0.485 
    im[1] += 0.456 
    im[2] += 0.406
    im = np.moveaxis(im, 0, 2)
    im *= 255
    im = np.clip(im,0,255)
    return (im).astype(np.uint8)
    
# Pretrained VGG16 model
vgg16 = resnet50(pretrained=True)
vgg16.eval() # disable dropout, batchnorm
SoftmaxWithXent = nn.CrossEntropyLoss()
print (".. loaded pre-trained vgg16")
xs, y_trues, y_preds, noises, y_preds_adversarial = [], [], [], [], []

for imloc in (image_location_generator("./images/")): 
    print(imloc)
    x = Variable(image_loader(imloc), requires_grad=True)
    output = vgg16.forward(x)
    y = Variable(torch.LongTensor(np.array([output.data.numpy().argmax()])), requires_grad = False)
    loss = SoftmaxWithXent(output, y)
    loss.backward()

    # Add perturbation 
    epsilon = 4.5 / 0.229 / 255
    x_grad  = torch.sign(x.grad.data)
    adv_x  = x.data + epsilon * x_grad  # we do not know the min/max because of torch's own stuff

    # Check adversarilized output 
    pred = np.argsort(-vgg16.forward(Variable(adv_x)).data.numpy())
    top3 = (pred)
    print(top3[0][:3])
    
    y_pred_adversarial = ImageNet_mapping[ str(np.argmax(vgg16.forward(Variable(adv_x)).data.numpy())) ]
    y_true = ImageNet_mapping[ str( int( y.data.numpy() ) ) ]
    
    if y_pred_adversarial == y_true:
        print ("Error: Could not adversarialize image ")
    else:
        xs.append(x.data.numpy())
        y_preds.append( y_true )
        y_trues.append( y_true )
        noises.append((adv_x - x.data).numpy())
        y_preds_adversarial.append( y_pred_adversarial )
        print (y_preds[-1], " | ", y_preds_adversarial[-1])

    x_ori=(tensor2im(x.data.numpy()))
    x_adv=(tensor2im(adv_x.numpy()))
    outfile = ('output/'+imloc.split('/')[2])
    print(np.max(np.abs(x_ori.astype(int)-x_adv.astype(int))))
    array2PIL(x_adv,outfile)
    if imloc == "./images/000.png":
    break
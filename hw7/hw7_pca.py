import os
import sys
import numpy as np 
from PIL import Image
from skimage.io import imread, imsave


def process(M): 
    M -= np.min(M)
    M /= np.max(M)
    M = (M * 255).astype(np.uint8)
    return M

def plot_average(mean):
    # Report 1
    average = process(mean)
    imsave('average.jpg', average.reshape(img_shape))  

def plot_eigenface(u,num):
    # Report 1.b
    for x in range(num):
        eigenface = process(u[:,x])
        imsave(str(x) + '_eigenface.jpg', eigenface.reshape(img_shape)) 
        
def compute_ratio(num):
    ## Report 1.d
    for i in range(num):
        number = s[i] * 100 / sum(s)
        print(number)
        
def main(argv):
    assert(len(argv)==4)
    IMAGE_PATH = argv[1]
    # Images for compression & reconstruction
    test_image = [argv[2]]
    out_image = [argv[3]]

    # Number of principal components used
    k = 5
    filelist = os.listdir(IMAGE_PATH)
    
    img_data = []
    for filename in filelist:
        try:
            path = IMAGE_PATH+filename
            tmp = imread(path)  
            img_shape = tmp.shape
            img_data.append(tmp.flatten())
        except:
            pass
        
    print("totol_image_num ", len(img_data))


    training_data = np.array(img_data).astype('float32')
    # Calculate mean & Normalize
    mean = np.mean(training_data, axis = 0)  
    training_data -= mean 

    # Use SVD to find the eigenvectors 
    u, s, v = np.linalg.svd(training_data.T, full_matrices = False)  

    for x, outfile in zip(test_image, out_image): 
        # Load image & Normalize
        picked_img = imread(IMAGE_PATH+x)
        X = picked_img.flatten().astype('float32') 
        X -= mean

        weight =  np.dot(X, u[:,:k])

        # Reconstruction
        reconstruct = process(weight.dot(u[:,:k].T) + mean)
        imsave(outfile, reconstruct.reshape(img_shape)) 
        
if __name__ == '__main__':
    main(sys.argv)
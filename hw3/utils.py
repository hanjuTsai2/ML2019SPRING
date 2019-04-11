import pandas as pd
import numpy as np
from keras.utils import to_categorical

def LoadData(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    Y_train = np.array(train['label'])
    Y_train = to_categorical(Y_train)
    
    X = np.array(train['feature'])
    X_train = []
    for i in range(X.shape[0]):
        x = np.array(X[i].split(' '))
        x = x.astype(np.int)
        X_train.append(x)
    X_train = np.array(X_train)
    X_train = X_train.reshape(X_train.shape[0],48,48,1)/255
    
    X = np.array(test['feature'])
    X_test = []
    for i in range(X.shape[0]):
        x = np.array(X[i].split(' '))
        x = x.astype(np.int)
        X_test.append(x)
    X_test = np.array(X_test)
    X_test = X_test.reshape(X_test.shape[0],48,48,1)/255
    
    return X_train, Y_train, X_test

class PARAM:
    input_shape = (48,48,1)
    epochs = 500
    batch_size = 128
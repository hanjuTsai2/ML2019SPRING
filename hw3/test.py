import numpy as np
import pandas as pd
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
from keras.backend import tensorflow_backend
from keras.layers.core import K
import sys

K.set_learning_phase(0)
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

def LoadData(path):
    test = pd.read_csv(path)

    X = np.array(test['feature'])
    X_test = []
    for i in range(X.shape[0]):
        x = np.array(X[i].split(' '))
        x = x.astype(np.int)
        X_test.append(x)
    X_test = np.array(X_test)
    X_test = X_test.reshape(X_test.shape[0],48,48,1)/255
    
    return X_test

arg = sys.argv
X_test = LoadData(arg[1])
outputFile = arg[2]

modellist = ['./model/0.h5', './model/8.h5', 'model/model27.h5', 
              'model/model28.h5', 'model/model30.h5', 'model/model33.h5', 'model/model31.h5']
weight = [ 0.67, 0.67, 0.6853, 0.68681 , 0.66, 0.679, 0.672]
xss = X_test

final = np.array([0.0] * xss.shape[0] * 7).reshape(xss.shape[0],7)
for i in range(len(modellist)):
    model = load_model(modellist[i])
    predict = model.predict(xss)
    final += predict * (weight[i])
    del model

for cnt, i in enumerate(final):
    final[cnt,:] = (i) / sum(i)
    
final[:,4] -= 0.08497144288577155
final[:,6] -= 0.08877867735470943

Y_pred = np.array([])
for i in range(len(final)):
    Y_pred = np.append(Y_pred, int(np.argmax(final[i])))

ans = []
for i in range(len(Y_pred)):
    ans.append([i,int(Y_pred[i])])
    
ans = pd.DataFrame(ans,columns=['id', 'label'])
ans.to_csv(outputFile , index=False)
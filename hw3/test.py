import numpy as np
import pandas as pd
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
from keras.backend import tensorflow_backend

from keras.layers.core import K
# K.set_learning_phase(0)

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

def LoadData():
    test = pd.read_csv('./data/test.csv')    
    X = np.array(test['feature'])
    X_test = []
    for i in range(X.shape[0]):
        x = np.array(X[i].split(' '))
        x = x.astype(np.int)
        X_test.append(x)
    X_test = np.array(X_test)
    X_test = X_test.reshape(X_test.shape[0],48,48,1)/255
    
    return X_test

X_test = LoadData()



# modellist = ['model/model17-0.66.h5', 'model/model23.h5', 'model/model21-0.6853']
modellist = ['./bag/0.h5', './bag/8.h5','model/model21-0.6853.h5']
weight = [0.67, 0.67, 0.6853,] #[0.66 , 0.672, 0.662]
# weight = [0.6820575627679119,
#          0.6725045926572353,
#          0.6750765464018257,
#          0.6754439680472665,
#          0.6756889159739273,
#          0.663686466596642,
#          0.6813065819177938,
#          0.6777709738359959,
#          0.6860992041162677,
#          0.6797353590682744]

# modellist = []
# for i in range(10):
#     modellist.append("./bag/" + str(i) + ".h5")
# print(modellist)

final = np.array([0.0] * 7178 * 7).reshape(7178,7)

outfile = "vote4"

for i in range(2,3):
    model = load_model(modellist[i])
    predict = model.predict(X_test, batch_size=128)
    final += predict * weight[i]

final_class = []
for i in range(len(final)):
    final_class.append(np.argmax(final[i]))

ans = []
for i in range(len(final_class)):
    ans.append([i,final_class[i]])

ans = pd.DataFrame(ans,columns=['id', 'label'])
ans.to_csv('data/'+ outfile +'.csv', index=False)
#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pickle
import pandas as pd
import numpy as np
    
class standardScaler():
    def fit(self, xss):
        self.mean = np.mean(xss, axis=0)
        self.sd = np.std(xss, axis=0)

    def transform(self, xss):
        xss = (xss-self.mean)/(self.sd)
        return(xss)
    

with open('ans/simple.pkl', 'rb') as f:
    dic = pickle.load(f)
    print(dic)

w = dic['w']
scaler = dic['scaler']

test = pd.read_csv('data/test.csv', encoding='Big5', header=None)
test = test.replace('NR', '0')
ans = pd.read_csv('data/sampleSubmission.csv', encoding='Big5')
test_feature = 18
total_test = []
row, col = test.shape
test_number = int(row/test_feature)
print(test_number)
for i in range(test_number):
    df = test.iloc[i*test_feature:(i+1)*test_feature, 2:]
    xs = df.values.ravel().astype(np.float)
    xs = (scaler.transform([xs])[0])
    xs = np.concatenate(([1], xs))
    val = np.dot(xs,w)
    val = max(val,0)
    ans.iloc[i,1] = val
ans.to_csv('ans/simple.csv',index=False)


# In[ ]:





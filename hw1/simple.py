import pickle
import pandas as pd
import numpy as np
import sys

class standardScaler():
    def fit(self, xss):
        self.mean = np.mean(xss, axis=0)
        self.sd = np.std(xss, axis=0)

    def transform(self, xss):
        xss = (xss-self.mean)/(self.sd)
        return(xss)

def main(argv):
    if(len(argv) < 3):
        print('python3 simple.py [input] [output]')
        return

    with open('ans/simple.pkl', 'rb') as f:
        dic = pickle.load(f)

    inputFile = argv[1]
    outputFile = argv[2]

    w = dic['w']
    scaler = dic['scaler']

    test = pd.read_csv( inputFile , encoding='Big5', header=None)
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
    ans.to_csv(outputFile ,index=False)


if __name__ =="__main__":
    main(sys.argv)  


    


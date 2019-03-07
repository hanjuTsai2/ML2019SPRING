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

def polyTransform(xss):
    xss = np.column_stack([xss, xss**2])
    return(xss)

def main(argv):
    if(len(argv) < 3):
        print('python3 simple.py [input] [output]')
        return
  
    ans = pd.read_csv('data/sampleSubmission.csv', encoding='Big5')
    
    ## read the test csv
    test = pd.read_csv( inputFile , encoding='Big5', header=None)
    test = test.replace('NR', '0')
    test_feature = 18
    total_test = []
    row, col = test.shape
    test_number = int(row/test_feature)

    ## load the model
    model2 = np.load('model.npy') 
    w2 = model2.item().get('w')
    scaler2 = model2.item().get('scaler')
    selected_var2 = model2.item().get('selected_var')

    ## count the value
    for i in range(test_number):
        df = test.iloc[i*test_feature:(i+1)*test_feature, 2:]
        tmp = df.values.ravel().astype(np.float)    
        xs = (scaler2.transform([tmp])[0])
        xs = np.array([[xs[i] for i in range(len(xs)) if i in selected_var2]])
        xs = polyTransform(xs)[0]
        xs = np.concatenate(([1], xs))
        val = np.dot(xs,w2)
        val = round(val,0)
        ans.iloc[i,1] = val 

    ans.to_csv(outputFile ,index=False)


if __name__ == "__main__":
    main(sys.argv)  



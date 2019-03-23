import sys
import numpy as np
import pandas as pd

class Scaler():
    def fit(self, xss, dummy_columns, numeric_columns):
        
        self.numeric_columns = numeric_columns
        self.dummy_columns = dummy_columns
        
        ## process numeric matrix
        numeric_vector = xss.iloc[:,numeric_columns]
        self.mean = np.mean(numeric_vector)
        self.std = np.std(numeric_vector)
        
        ## process dummy matrix
        dummy_vector = xss.iloc[:,dummy_columns]
        self.proportion = np.mean(dummy_vector)
        
    def transform_dummy(self,xs,proportion):
        trans_xs = [1 - proportion if x == 1 else proportion for x in xs]
        return trans_xs
        
    def transform_numeric(self,xs, mean, std):
        xs = (xs-mean)/std
        return(xs)

    def transform(self, xss):
        row, col = xss.shape
        df = []
        for c in range(col):
            xs = xss.iloc[:,c]
            if c in self.dummy_columns:
                dff = xs
            else:
                idx = self.numeric_columns.index(c)
                mean = self.mean[idx]
                std = self.std[idx]
                dff = self.transform_numeric(xs,mean,std)
                
            df.append(dff)
        df = (np.column_stack(df))
        return(pd.DataFrame(df))

def sigmoid(predict):
    return(1/(1 + np.exp(-predict)))

        
def LoadData(X_testFile): 
    X_test = pd.read_csv(X_testFile)
    return  X_test

def main():    
    ## load model
    params = np.load("log.npy")
    w = params.item().get('w')
    scaler = params.item().get('scaler')

    arg = sys.argv
    X_test = LoadData(arg[1])
    outputFile = arg[2]

     ## define dummy and continuous number
    numeric_columns = [0, 1, 3, 4, 5]
    dummy_columns = list(set(range(X_test.shape[1])) - set(numeric_columns))
    
    ## transform the data
    X_test = scaler.transform(X_test)

    ## output testing data
    row , col = X_test.shape
    ans = []
    for i in range((row)):
        x = X_test.iloc[i,:]
        x = np.concatenate(([1] ,x))
        predict = np.dot(x,w)
        predict = sigmoid(predict)
        predict =  predict = 0 if predict < 0.5 else 1
        val = predict
        ans.append([i+1 ,val])

    ans = pd.DataFrame(ans,columns=['id', 'label'])
    print(np.sum(ans['label']))
    ans.to_csv(outputFile, index=False)
        
if __name__ == "__main__":
    main()
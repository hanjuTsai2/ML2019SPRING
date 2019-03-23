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

def gaussian(xs, mu, cov):
    col = len(xs)
    coeff = 1 / np.sqrt((2 * np.pi) ** col)
    determinant = 1 / np.sqrt(np.linalg.det(cov))
    term = (-1/2) * np.dot((xs-mu).T, np.linalg.inv(cov))
    term = np.dot(term,(xs-mu))
    
    prob = coeff * determinant * np.exp(term)
    return(prob)
        
def LoadData(X_testFile): 
    X_test = pd.read_csv(X_testFile)
    return  X_test

def main():    
    ## load model
    params = np.load("gen.npy")
    mu_0 = params.item().get('mu_0')
    mu_1 = params.item().get('mu_1')
    cov = params.item().get('cov')

    arg = sys.argv
    X_test = LoadData(arg[1])
    outputFile = arg[2]

     ## define dummy and continuous number
    numeric_columns = [0, 1, 3, 4, 5]
    dummy_columns = list(set(range(X_test.shape[1])) - set(numeric_columns))

    ## output testing data
    ans = []
    for i in range(len(X_test)):
        x = X_test.iloc[i,:]
        c0 = gaussian(x, mu_0, cov)
        c1 = gaussian(x, mu_1, cov)
        value = 0 if c0 > c1 else 1
        ans.append([i+1 ,value])

    ans = pd.DataFrame(ans,columns=['id', 'label'])
    ans.to_csv(outputFile, index=False)
    print("Save result to ..." + outputFile)
        
if __name__ == "__main__":
    main()
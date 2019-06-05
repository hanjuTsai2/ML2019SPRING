import os
import sys
import pandas as pd
import numpy as np
import torch
from model import Classifier
from mydataset import  MyDataset
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset

torch.manual_seed(0)

def LoadData(test_path):
    test = pd.read_csv(test_path)
    X = np.array(test['feature'])
    X_test = []
    for i in range(X.shape[0]):
        x = np.array(X[i].split(' '))
        x = x.astype(np.int)
        X_test.append(x)
    X_test = np.array(X_test)
    X_test = X_test.reshape(X_test.shape[0],48,48,1).astype(np.uint8)
    
    return X_test

def saving_compression(net):
    best_model_wts = net.state_dict()
    for key, value in best_model_wts.items():
        path = os.path.join('static_dict', key)
        compression = value.cpu().numpy()
        if 'num_batches_tracked' in key:
            np.savez_compressed(path, key=compression)
            continue
        compression = np.around(compression * 8192).astype('int16') #s2-13
        np.savez_compressed(path, key=compression)

def loading_compression(net):
    dd = net.state_dict()
    for key, value in dd.items():
        path = os.path.join('static_dict', key)
        if 'num_batches_tracked' in key:
            compression = np.load(path + '.npz')['key']
            dd[key] = torch.from_numpy(compression)
            continue
        compression = np.load(path+'.npz')['key'].astype('float32') / 8192
        compression = compression.astype('float32')
        dd[key] = torch.from_numpy(np.array(compression))
    net.load_state_dict(dd)
    return net

def main():
    x_test = LoadData(sys.argv[1])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(44),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0],
                             std=[255])
    ])
    batch_size = 128
    test_set = MyDataset(x_test, None, val_transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = Classifier().to(device)
    model = loading_compression(model)

    ans = np.array([])
    for i, data in enumerate(test_loader):
        pred = model(data.to(device))
        ans = np.append(ans,np.argmax(pred.data.cpu().numpy(), axis=1))
    
    ans = ans.astype(int)
    df = [[cnt,i] for cnt, i in enumerate(ans)]
    df = pd.DataFrame(df, columns=['id', 'label'])
    df.to_csv(sys.argv[2], index=None)

if __name__ == "__main__":
    main()

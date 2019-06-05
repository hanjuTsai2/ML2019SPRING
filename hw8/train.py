import sys
import csv
import time
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from keras.utils import to_categorical
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

import os
from model import Classifier
from mydataset import MyDataset

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

def LoadData(train_path):
    train = pd.read_csv(train_path)
    Y_train = np.array(train['label'])

    X = np.array(train['feature'])
    X_train = []
    for i in range(X.shape[0]):
        x = np.array(X[i].split(' '))
        x = x.astype(np.int)
        X_train.append(x)
    X_train = np.array(X_train)
    X_train = X_train.reshape(X_train.shape[0],48,48,1).astype(np.uint8)
    
    ## validation
    x_train, val_data, x_label, val_label = train_test_split(X_train, Y_train, test_size=0.2) 
    return x_train, val_data, x_label, val_label

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = 0.001 * (0.1 ** (epoch // 200))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    x_train, val_data, x_label, val_label = LoadData(sys.argv[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(44),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0],
                             std=[255])
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(44),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0],
                             std=[255])
    ])

    train_set = MyDataset(x_train, x_label, data_transform)
    val_set = MyDataset(val_data, val_label, val_transform)

    batch_size = 128
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8)

    model = Classifier().to(device)
    model.initialize_weights()
    print(model.eval())
    
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    best_acc = 0.0
    num_epoch = 1000

    for epoch in range(num_epoch):
        adjust_learning_rate(optimizer, epoch)
        
        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        model.train()
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            train_pred = model(data[0].to(device))
            batch_loss = loss(train_pred, data[1].to(device))
            batch_loss.backward()
            optimizer.step()

            train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            train_loss += batch_loss.item()

            progress = ('#' * int(float(i)/len(train_loader)*40)).ljust(40)
            print ('[%03d/%03d] %2.2f sec(s) | %s |' % (epoch+1, num_epoch, \
                    (time.time() - epoch_start_time), progress), end='\r', flush=True)
        
        model.eval()
        for i, data in enumerate(val_loader):
            val_pred = model(data[0].cuda())
            batch_loss = loss(val_pred, data[1].to(device))

            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()

            progress = ('#' * int(float(i)/len(val_loader)*40)).ljust(40)
            print ('[%03d/%03d] %2.2f sec(s) | %s |' % (epoch+1, num_epoch, \
                    (time.time() - epoch_start_time), progress), end='\r', flush=True)

        val_acc = val_acc/val_set.__len__()
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
                (epoch + 1, num_epoch, time.time()-epoch_start_time, \
                train_acc/train_set.__len__(), train_loss, val_acc, val_loss))
        
        # scheduler.step(val_loss)
        if (val_acc > best_acc):
            best_acc = val_acc
            
            if not os.path.exists("static_dict/"):
                os.system('mkdir static_dict/')
                
            saving_compression(model)
            os.system("du static_dict/ --apparent-size --bytes --max-depth=0")
            with open('acc.txt','w') as f:
                f.write(str(epoch)+'\t'+str(val_acc)+'\t')
                    
#             torch.save(model.state_dict(), 'save/model.pth')
            print ('Model Saved!')

if __name__ == "__main__":
    main()
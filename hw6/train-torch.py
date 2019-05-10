## Import Package

# standard library
import os
import csv
import sys
import argparse
from multiprocessing import Pool

# optional library
import jieba
import pandas as pd
import numpy as np
import emoji
import re
from gensim.models import Word2Vec

# pytorch library
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable 
torch.nn.Module.dump_patches = True

from torch.utils.data import TensorDataset, DataLoader
# from sklearn.model_selection import train_test_split, KFold

add_word = ['好不好', '不就', '要不要', '笑死', '頂大', '有愛','乾你屁事', '不懂' ,'屌打', 'ㄏㄏ']
for word in add_word:
    jieba.suggest_freq(word, True)
    
class Preprocess():
    def __init__(self, data_dir, label_dir, args):
        # Load jieba library
        jieba.load_userdict(args.jieba_lib)
        self.embed_dim = args.word_dim
        self.seq_len = args.seq_len
        self.wndw_size = args.wndw
        self.word_cnt = args.cnt
        self.save_name = 'word2vec'
        self.index2word = []
        self.word2index = {}
        self.vectors = []
        self.unk = "<UNK>"
        self.pad = "<PAD>"
        # Load corpus
        if data_dir != None:
            # Read data
            dm = pd.read_csv(data_dir)
            data = dm['comment']
            # Tokenize with multiprocessing
            # List in list out with same order
            # Multiple workers
            P = Pool(processes=4) 
            data = P.map(self.tokenize, data)
            P.close()
            P.join()
            self.data = data
            self.data = self.data[:119017]
#             self.data = self.clean_data(self.data)
            
        if label_dir != None:
            # Read Label
            dm = pd.read_csv(label_dir)
            self.label = [int(i) for i in dm['label']]
            self.label = self.label[:119017]
            
    def clean_data(self, data):
#         data = [ [w for w in sen if w not in emoji.UNICODE_EMOJI] for sen in data]
        data = [ [w for w in sen if not re.search(r"[a-zA-Z0-9]", w)] for sen in data]
        print(data[0])
        return data
    
    def tokenize(self, sentence):
        # TODO
        tokens = jieba.cut(sentence)
        tokens = ' '.join(tokens).split()
        return tokens

    def get_embedding(self, load=False):
        print("=== Get embedding")
        # Get Word2vec word embedding
        if load:
            embed = Word2Vec.load(self.save_name)
        else:
            test_data = pd.read_csv(args.test_X)['comment']
            P = Pool(processes=4) 
            test_data = P.map(self.tokenize, test_data)
            P.close()
            P.join()
            embed = Word2Vec(self.data+test_data,
                             size=self.embed_dim
                             ,window=self.wndw_size,
                             min_count=self.word_cnt, iter=100, workers=8)
            embed.save(self.save_name)
            
        # Create word2index dictinonary
        # Create index2word list
        # Create word vector list
        
        for i, word in enumerate(embed.wv.vocab):
            print('=== get words #{}'.format(i+1), end='\r')
            self.word2index[word] = len(self.word2index)
            self.index2word.append(word)
            self.vectors.append(embed[word])
        
        self.vectors = torch.tensor(self.vectors)
        # Add special tokens
        self.add_embedding(self.pad)
        self.add_embedding(self.unk)
        print("=== total words: {}".format(len(self.vectors)))
        return self.vectors

    def add_embedding(self, word):
        # Add random uniform vector
        vector = torch.empty(1, self.embed_dim)
        torch.nn.init.uniform_(vector)
        self.word2index[word] = len(self.word2index)
        self.index2word.append(word)
        self.vectors = torch.cat([self.vectors, vector], 0)

    def get_indices(self,test=False):
        all_indices = []
        # Use tokenized data
        for i, sentence in enumerate(self.data):
            print('=== sentence count #{}'.format(i+1), end='\r')
            sentence_indices = []
            for word in sentence:
                # if word in word2index append word index into sentence_indices
                # if word not in word2index append unk index into sentence_indices
                try:
                    sentence_indices.append(self.word2index[word])
                except:
                    sentence_indices.append(self.word2index[self.unk])
                    
            # pad all sentence to fixed length
            sentence_indices = self.pad_to_len(sentence_indices, self.seq_len, self.word2index[self.pad])
            all_indices.append(sentence_indices)
            
        if test:
            return torch.LongTensor(all_indices)         
        else:
            return torch.LongTensor(all_indices), torch.LongTensor(self.label)        

    def pad_to_len(self, arr, padded_len, padding=0):
        seq_len = len(arr)
        if seq_len < padded_len:
            arr += [padding] * (padded_len-seq_len)
        elif seq_len > padded_len:
            arr = arr[:padded_len]
        return arr

class LSTM_Net(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.5, fix_emb=True):
        super(LSTM_Net, self).__init__()
        # Create embedding layer
        self.embedding = torch.nn.Embedding(embedding.size(0),embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)
        # Fix/Train embedding 
        self.embedding.weight.requires_grad = False if fix_emb else True
        self.embedding_dim = embedding.size(1)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)
        
        self.classifier = nn.Sequential(
                            nn.Dropout(dropout),
                            nn.Linear(hidden_dim*2, 1),
                            nn.Sigmoid())

    def forward(self, inputs):
        #torch.Size([128, 50])
        inputs = self.embedding(inputs)
        # torch.Size([128, 50, 100]) 
        x, _ = self.lstm(inputs, None)
        # torch.Size([128, 50, 300])
        x = x[:, -1, :]
        # torch.Size([128, 300])
        x = self.classifier(x) 
        # torch.Size([128, 1])
        return x

    
def evaluation(outputs, labels):
    outputs[outputs >= 0.5] = 1
    outputs[outputs < 0.5] = 0
    correct = torch.sum(torch.eq(outputs, labels)).item()
    return correct

def training(args, train, valid, model, device, model_name):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\n=== start training, parameter total:{}, trainable:{}'.format(total, trainable))
    model.train()
    batch_size, n_epoch = args.batch, args.epoch
    criterion = nn.BCELoss()
    t_batch = len(train) 
    v_batch = len(valid) 

    optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    
    total_loss, total_acc, best_acc = 0, 0, 0
    for epoch in range(n_epoch):
        total_loss, total_acc = 0, 0
        # training set
        for i, (inputs, labels) in enumerate(train):
            inputs = inputs.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.float)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            correct = evaluation(outputs, labels)
            total_acc += (correct / batch_size)
            total_loss += loss.item()
            
            print('[ Epoch{} == {}/{} ] loss:{:.3f} acc:{:.3f} '.format(
            	epoch+1, i+1, t_batch, loss.item(), correct*100 / batch_size), end='\r')
        print('\nTrain | Loss:{:.5f} Acc: {:.3f} '.format(total_loss / t_batch, total_acc / t_batch*100))

        # validation set
        model.eval()
        with torch.no_grad():
            total_loss, total_acc = 0, 0
            for i, (inputs, labels) in enumerate(valid):
                inputs = inputs.to(device, dtype=torch.long)
                labels = labels.to(device, dtype=torch.float)
                outputs = model(inputs)
                outputs = outputs.squeeze()
                loss = criterion(outputs, labels)
                correct = evaluation(outputs, labels)
                total_acc += (correct / batch_size)
                total_loss += loss.item()
                
            print("Valid | Loss:{:.5f} Acc: {:.3f} ".format(total_loss/v_batch, total_acc/v_batch*100))
            if total_acc > best_acc:
                best_acc = total_acc
                if total_acc / v_batch * 100 > 75:
                    torch.save(model, "./mod_{}_ckpt_{:.3f}".format(model_name, total_acc/v_batch*100))
                    print('saving model with acc {:.3f}'.format(total_acc/v_batch*100))
                    
        model.train()
    return best_acc/v_batch

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess = Preprocess(args.train_X, args.train_Y, args)
    # Get word embedding vectors
    embedding = preprocess.get_embedding(load=False)
    
    # Get word indices
    data, label = preprocess.get_indices()
    
    train_num = data.numpy().shape[0] // 10 * 9
    X_train, X_test = data.numpy()[:train_num], data.numpy()[train_num:]
    y_train, y_test = label.numpy()[:train_num], label.numpy()[train_num:]
    X_train = torch.tensor(X_train)
    X_test = torch.tensor(X_test)
    y_train = torch.tensor(y_train)
    y_test = torch.tensor(y_test)

    # Split train and validation set and create data loader
    dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(dataset, batch_size=args.batch)

    dataset = TensorDataset(X_test, y_test)
    val_loader = DataLoader(dataset, batch_size=args.batch)

    # Get model
    model = LSTM_Net(embedding, args.word_dim, args.hidden_dim, args.num_layers)
    model = model.to(device)

    # Start training
    best_acc = training(args, train_loader, val_loader, model, device, 'best')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
#     parser.add_argument('model_dir', type=str, help='[Output] Your model checkpoint directory')
    parser.add_argument('train_X',type=str, help='[Input] Your train_x.csv')
    parser.add_argument('train_Y',type=str, help='[Input] Your train_y.csv')
    parser.add_argument('test_X',type=str, help='[Input] Your train_y.csv')
    parser.add_argument('jieba_lib',type=str, help='[Input] Your jieba dict.txt.big')

    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--batch', default=128, type=int)
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--seq_len', default=30, type=int)
    parser.add_argument('--word_dim', default=100, type=int)
    parser.add_argument('--hidden_dim', default=100, type=int)
    parser.add_argument('--wndw', default=3, type=int)
    parser.add_argument('--cnt', default=3, type=int)
    args = parser.parse_args()
    main(args)


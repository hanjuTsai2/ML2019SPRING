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
from gensim.models import Word2Vec

# pytorch library
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.nn.Module.dump_patches = True

from torch.utils.data import TensorDataset, DataLoader

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
            
        if label_dir!=None:
            # Read Label
            dm = pd.read_csv(label_dir)
            self.label = [int(i) for i in dm['label']]
            self.label = self.label[:119017]

    def tokenize(self, sentence):
        tokens = jieba.cut(sentence)
        tokens = ' '.join(tokens).split()
        return tokens

    def get_embedding(self, load=False):
        print("=== Get embedding")
        # Get Word2vec word embedding
        if load:
            embed = Word2Vec.load(self.save_name)
        else:
            embed = Word2Vec(self.data, size=self.embed_dim, window=self.wndw_size, min_count=self.word_cnt, iter=100, workers=8)
            embed.save(self.save_name)
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
        for i, sentence in enumerate(self.data):
            print('=== sentence count #{}'.format(i+1), end='\r')
            sentence_indices = []
            for word in sentence:
                try:
                    sentence_indices.append(self.word2index[word])
                except:
                    sentence_indices.append(self.word2index[self.unk])

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


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess = Preprocess(args.test_X, None, args)
    
    model_dir_name = 'best_bag_0_1/'
    # Get word embedding vectors
    embedding = preprocess.get_embedding(load=True)
    model_dirs = os.listdir(model_dir_name)
    
    # Get word indices
    test_x = preprocess.get_indices(test=True)
    test_x = test_x.to(device, dtype=torch.long)
    
    dataset = TensorDataset(test_x)
    test_loader = DataLoader(dataset, batch_size=args.batch)
    
    total_y_pred = np.array([0.0] * len(test_x))
    model_num = 0
    for model_dir in model_dirs:
        try:
            model_name = os.path.join(model_dir_name, model_dir)
            print(model_name)
            model = torch.load(model_name)
            y_pred = np.array([]*len(test_loader))
            for cnt, test_x in enumerate(test_loader):
                test_x = test_x[0].to(device, dtype=torch.long)
                outputs = model(test_x)
                outputs = outputs.squeeze()
                pred = (outputs.cpu().detach().numpy())
                y_pred = np.append(y_pred,pred)

            print(y_pred)
            y_pred = y_pred.ravel()
            model_num += 1
            print(model_num)
            total_y_pred += y_pred 
            del model   
        except:
            print('broken', model_dir)
            
    print('cnt', model_num)
    total_y_pred /= model_num
    print(total_y_pred)
    
    df = []
    for cnt,i in enumerate(total_y_pred):
        i = 1 if i >= 0.5 else 0
        df.append([cnt,i])
    ans = pd.DataFrame(df,columns=['id','label'])
    ans.to_csv(args.output, index=None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('jieba_lib',type=str, help='[Input] Your jieba dict.txt.big')
    parser.add_argument('test_X',type=str, help='[Input] Your train_x.csv')
    parser.add_argument('output',type=str, help='[Input] Your output.csv')

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


import numpy as np
import pandas as pd

import jieba
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

train_x_file = 'data/train_x.csv'
train_y_file = 'data/train_y.csv'
test_x_file = 'data/test_x.csv'

x_train = pd.read_csv(train_x_file).values
y_train = pd.read_csv(train_y_file).values
x_test = pd.read_csv(test_x_file).values

seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造") 
print(", ".join(seg_list))
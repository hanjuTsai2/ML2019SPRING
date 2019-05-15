import pandas as pd
import numpy as np
import sys

ans1=pd.read_csv(sys.argv[1])
ans2=pd.read_csv(sys.argv[2])

print('total', ans1.shape[0])
print('diff_num',np.sum(ans1.label!=ans2.label))
print(np.mean(ans1.label==ans2.label))

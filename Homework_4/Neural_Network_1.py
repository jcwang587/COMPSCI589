import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Load data
df = pd.read_csv('hw3_wine.csv', sep='\t')
col_class = df.pop('# class')
df.insert(len(df.columns), '# class', col_class)
col_mean = df.mean().tolist()
for idx in range(0, len(df.columns) - 1):
    df.loc[df[df.keys()[idx]] <= col_mean[idx], df.keys()[idx]] = 0
    df.loc[df[df.keys()[idx]] > col_mean[idx], df.keys()[idx]] = 1
# Split the original dataset
list_target = df['# class'].unique()
df2 = df[df['# class'].isin([list_target[0]])]
df1 = df[df['# class'].isin([list_target[1]])]
df0 = df[df['# class'].isin([list_target[2]])]
# Split into folds
kfold = []
fold_size2 = int(len(df2) / 10)
fold_size1 = int(len(df1) / 10)
fold_size0 = int(len(df0) / 10)
for k in range(0, 9):
    fold2 = df2.sample(n=fold_size2)
    fold1 = fold2.append(df1.sample(n=fold_size1))
    fold0 = fold1.append(df0.sample(n=fold_size0))
    df2 = df2[~df2.index.isin(fold2.index)]
    df1 = df1[~df1.index.isin(fold1.index)]
    df0 = df0[~df0.index.isin(fold0.index)]
    kfold.append(fold0)
kfold.append(df2.append(df1.append(df0)))

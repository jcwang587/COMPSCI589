import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from collections import Counter

# Load the dataset
df = pd.read_csv('house_votes_84.csv')

# Shuffle the dataset
df_sf = shuffle(df)
X = df_sf[df.columns[0:15]]
y = df_sf[df.columns[16]]

# Randomly partition the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Normalize the dataset
X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
X_test = (X_test - X_test.min()) / (X_test.max() - X_test.min())

import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def calculate_prob_entropy(data):
    prob = np.array(pd.Series(data).value_counts(normalize=True))
    entropy = sum(-prob * np.log2(prob))
    return entropy


def find_branch(xdata, ydata):
    # If there are no more attributes that can be tested
    if len(xdata.columns) == 1:
        branch = pd.Series(xdata).value_counts().idxmax()

    # If all instances belong to the same class
    elif pd.Series(ydata).value_counts().values[0] == len(ydata):
        branch = ydata[0]

    else:
        # Calculate entropy of the original dataset
        i_data = calculate_prob_entropy(ydata)

        # Calculate entropy of the attributes
        i_attribute = {}
        for attribute in range(0, len(xdata.columns)):
            category_list = list(dict(pd.Series(xdata.iloc[:, attribute]).value_counts()).keys())
            i_category = {}
            # Calculate entropy of partitions resulting from test
            for category in category_list:
                i_category[category] = calculate_prob_entropy(ydata.loc[xdata.iloc[:, attribute] == category])
            # Calculate average entropy of the resulting partitions
            probability = pd.Series(xdata.iloc[:, attribute]).value_counts(normalize=True)
            i_attribute[attribute] = sum(probability.values * list(i_category.values()))

        # Calculate information gain
        i_gain = i_data - np.array(list(i_attribute.values()))

        # Decide the attribute
        best_attribute = np.where(i_gain == np.max(i_gain))[0][0]
        branch = xdata.columns[best_attribute]
    return branch


def split_data(xdata_input, ydata_input, branch):
    xdata_output0 = xdata_input[xdata_input[branch] == 0].drop(columns=branch)
    xdata_output05 = xdata_input[xdata_input[branch] == 0.5].drop(columns=branch)
    xdata_output1 = xdata_input[xdata_input[branch] == 1].drop(columns=branch)
    ydata_output0 = ydata_input[xdata_input[branch] == 0]
    ydata_output05 = ydata_input[xdata_input[branch] == 0.5]
    ydata_output1 = ydata_input[xdata_input[branch] == 1]
    return xdata_output0, xdata_output05, xdata_output1, ydata_output0, ydata_output05, ydata_output1


# Load the dataset
df = pd.read_csv('house_votes_84.csv')

# Shuffle the dataset
df_sf = shuffle(df)
X = df_sf[df.columns[0:16]]
y = df_sf[df.columns[16]]

# Randomly partition the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Normalize the dataset
X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
X_test = (X_test - X_test.min()) / (X_test.max() - X_test.min())

branch1 = find_branch(X_train, y_train)
#
# xdata_output_0 = split_data(X_train, y_train, branch1)[1]
# ydata_output_0 = split_data(X_train, y_train, branch1)[4]

decision_tree = {branch1: {}}
classList = [data[-1] for data in X_train]  # 获取数据集中所属类别的数据

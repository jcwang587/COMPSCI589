import utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # Load data
    data = utils.load_training_set(0.5, 0.3)

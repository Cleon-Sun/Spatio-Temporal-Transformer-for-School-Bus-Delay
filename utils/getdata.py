import pandas as pd
import numpy as np


def get_data():
    value = pd.read_csv('PeMS-M/V_228.csv', header=None).values
    adj = pd.read_csv('PeMS-M/W_228.csv', header=None).values

    train_data = value[:int(0.6 * len(value)), :]
    valid_data = value[int(0.6 * len(value)):int(0.8 * len(value)), :]
    test_data = value[int(0.8 * len(value)):, :]

    x_stats = {'mean': np.mean(train_data), 'std': np.std(train_data)}

    train_data = (train_data - x_stats['mean']) / x_stats['std']
    valid_data = (valid_data - x_stats['mean']) / x_stats['std']
    test_data = (test_data - x_stats['mean']) / x_stats['std']

    return train_data, valid_data, test_data, adj, x_stats

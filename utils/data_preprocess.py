import numpy as np
from datetime import datetime

def balanced_train_test_split(
    X: np.ndarray, Y: np.ndarray, unique_classes, test_split_ratio=0.3
):
    train_X = []
    train_Y = []
    test_X = []
    test_Y = []

    # Split for each class to ensure similar distribution
    for c in unique_classes:
        c_indexes = np.where(Y == c)
        X_where_c = X[c_indexes]
        targets = Y[c_indexes]
        test_data_count = int(test_split_ratio * len(X_where_c))
        split_index = len(X_where_c) - test_data_count

        train_X.extend(X_where_c[:split_index])
        test_X.extend(X_where_c[split_index:])
        train_Y.extend(targets[:split_index])
        test_Y.extend(targets[split_index:])

    return (np.array(train_X), np.array(train_Y), np.array(test_X), np.array(test_Y))


def train_test_split(X: np.ndarray, Y: np.ndarray, test_split_ratio=0.3, shuffle = True):
    index = np.arange(X.shape[0])

    if (shuffle == True):
        np.random.shuffle(index)
    
    dataset_X = X[index]
    dataset_Y = Y[index]

    test_data_count = int(test_split_ratio * X.shape[0])
    split_index = X.shape[0] - test_data_count

    train_X = dataset_X[:split_index]
    train_Y = dataset_Y[:split_index]
    test_X = dataset_X[split_index:]
    test_Y = dataset_Y[split_index:]

    return (train_X, train_Y, test_X, test_Y)

def to_seconds_since_midnight(datestr, format = '%Y-%m-%d %H:%M:%S'):
    dt = datetime.strptime(datestr, format)

    time_since_midnight = dt - dt.replace(hour=0, minute=0, second=0, microsecond=0)

    # Value is normalized by dividing with total seconds in a day
    seconds = time_since_midnight.total_seconds() / 86.400
    return round(float(seconds))

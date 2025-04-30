import numpy as np


def within_class_scatter_matrix(X: np.ndarray, Y: np.ndarray):
    classes = np.unique(Y)
    features_count = X.shape[1]
    Sw = np.zeros((features_count, features_count))

    for cl in classes:
        X_cl = X[Y == cl]

        mean_vec = np.mean(X_cl, axis=0)

        Sw += ((X_cl - mean_vec).T @ (X_cl - mean_vec))

    return Sw


def between_class_scatter_matrix(X: np.ndarray, Y: np.ndarray):
    classes = np.unique(Y)
    features_count = X.shape[1]
    mean_all_cl = np.mean(X, axis=0)

    Sb = np.zeros((features_count, features_count))

    for cl in classes:
        X_cl = X[Y == cl]
        mean_cl = np.mean(X_cl, axis=0)
        cl_samples_count = X_cl.shape[0]

        # Reshape to column vectors for outer product
        mean_diff = (mean_cl - mean_all_cl).reshape(features_count, 1)

        # Add to the between-class scatter matrix
        Sb += cl_samples_count * (mean_diff @ mean_diff.T)

    return Sb

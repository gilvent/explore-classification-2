import numpy as np


def within_class_scatter_matrix(X: np.ndarray, Y: np.ndarray):
    classes = np.unique(Y)
    features_count = X.shape[1]
    Sw = np.zeros((features_count, features_count))

    for cl in classes:
        X_cl = X[Y == cl]

        mean_vec = np.mean(X_cl, axis=0)

        Sw += (X_cl - mean_vec).T @ (X_cl - mean_vec)

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


def fld_vector(X: np.ndarray, Y: np.ndarray):
    """
    Fisher Linear Discriminant.

    FLD vector (along the column) gives a 1D projection direction that maximizes class separability for 2-class case.
    Based on the formula:

    w = inverse(Sw) @ (mean_1 - mean_0)
    """

    X_cl_1 = X[Y == 1]
    X_cl_0 = X[Y == 0]
    m1 = np.mean(X_cl_1, axis=0)
    m0 = np.mean(X_cl_0, axis=0)
    feature_count = X_cl_1.shape[1]

    # Reshape the mean vector into n_features row x 1 column
    mean_diff = (m1 - m0).reshape(feature_count, 1)

    Sw = within_class_scatter_matrix(X=X, Y=Y)
    Sw_inv = np.linalg.inv(Sw)
    w = Sw_inv @ mean_diff
    w = w / np.linalg.norm(w)

    return w

def trace_ratio(X, Y):
    """
    J(w) = tr{Sm} / tr {Sw} for n-dimension (before projection)

    J(w) = Sm / Sw for 1-dimension (after projection)

    Measures class separability on binary classification, where features are projected to 1-dimension
    """
    Sw = within_class_scatter_matrix(X=X, Y=Y)
    Sb = between_class_scatter_matrix(X=X, Y=Y)
    Sm = Sw + Sb

    return np.trace(Sm) / np.trace(Sw)


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


def fld_vector(within_cl_scatter, X_cl_1: np.ndarray, X_cl_0: np.ndarray):
    """
    Fisher Linear Discriminant.

    FLD vector (along the column) gives a 1D projection direction that maximizes class separability for 2-class case.
    Based on the formula:

    w = inverse(Sw) @ (mean_1 - mean_0)
    """
    m1 = np.mean(X_cl_1, axis=0)
    m0 = np.mean(X_cl_0, axis=0)
    feature_count = X_cl_1.shape[1]

    # Reshape the mean vector into n_features row x 1 column
    mean_diff = (m1 - m0).reshape(feature_count, 1)

    Sw_inv = np.linalg.inv(within_cl_scatter)
    w = Sw_inv @ mean_diff
    w = w / np.linalg.norm(w)

    return w


def fdr(X_cl_1_projected: np.ndarray, X_cl_0_projected: np.ndarray):
    """
    Fisher Discriminant Ratio.

    Used to measure class separability on 1d projection. Based on formula:

    FDR = (mean_1 - mean_2)^2 / variance_1 + variance_2

    Mean and variance, respectively, are mean values and variances of X in the two classes after projection
    """
    cl_1_projected_mean = np.mean(X_cl_1_projected, axis=0)
    cl_0_projected_mean = np.mean(X_cl_0_projected, axis=0)

    # Calculate class 1 variance
    cl_1_samples = X_cl_1_projected.shape[0]
    cl_1_mean_diff_matrix = X_cl_1_projected - cl_1_projected_mean
    # Variance = Sum of (projected x - projected x mean)^2 / number of samples
    cl_1_variance = np.sum(np.power(cl_1_mean_diff_matrix, 2)) / cl_1_samples

    # Calculate class 0 variance
    cl_0_samples = X_cl_0_projected.shape[0]
    cl_0_mean_diff_matrix = X_cl_0_projected - cl_0_projected_mean
    # Variance = Sum of (projected x - projected x mean)^2 / number of samples
    cl_0_variance = np.sum(np.power(cl_0_mean_diff_matrix, 2)) / cl_0_samples

    fdr = np.power(cl_1_projected_mean - cl_0_projected_mean, 2) / (
        cl_0_variance + cl_1_variance
    )
    return fdr.item()

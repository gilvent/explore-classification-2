import numpy as np

# w ∝ Sw^(-1) (mean1 - mean0)
def projection_vector(within_cl_scatter, X_cl_1: np.ndarray, X_cl_0: np.ndarray):
    m1 = np.mean(X_cl_1, axis=0)
    m0 = np.mean(X_cl_0, axis=0)
    feature_count = X_cl_1.shape[1]

    # Reshape the mean vector into n_features row x 1 column
    mean_diff = (m1 - m0).reshape(feature_count, 1)

    Sw_inv = np.linalg.inv(within_cl_scatter)
    w = Sw_inv @ mean_diff
    w = w / np.linalg.norm(w)

    return w

# Using Fisher's criterion formula
# J(w) = (w_transpose . (Sb . w)) / (w_transpose . (Sw . w))
def fdr(projection_vector: np.ndarray, within_cl_scatter: np.ndarray, between_cl_scatter: np.ndarray):
    w = projection_vector
    Sw = within_cl_scatter
    Sb = between_cl_scatter

    numerator = w.T @ (Sb @ w)
    denominator = w.T @ (Sw @ w)

    return (numerator / denominator).item()


def fdr_1d(X_cl_1_projected: np.ndarray, X_cl_0_projected: np.ndarray):
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

    fdr =  np.power(cl_1_projected_mean - cl_0_projected_mean, 2) / (cl_0_variance + cl_1_variance)
    return fdr.item()
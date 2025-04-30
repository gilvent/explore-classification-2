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
def class_separability(projection_vector: np.ndarray, within_cl_scatter: np.ndarray, between_cl_scatter: np.ndarray):
    w = projection_vector
    Sw = within_cl_scatter
    Sb = between_cl_scatter

    numerator = w.T @ (Sb @ w)
    denominator = w.T @ (Sw @ w)

    return (numerator / denominator).item()

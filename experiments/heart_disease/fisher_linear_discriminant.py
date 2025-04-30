import numpy as np
import matplotlib.pyplot as plt
from class_separability_measures.scatter_matrix import (
    within_class_scatter_matrix,
    between_class_scatter_matrix,
)
from class_separability_measures.fisher_linear_discriminant import (
    projection_vector,
    fdr,
    fdr_1d,
)


def main():
    dataset = np.loadtxt(
        fname="data/heart_disease_cleveland_processed.txt", delimiter=",", dtype=str
    )

    # Remove rows with missing values
    filtered_dataset = dataset[~np.char.equal(dataset, "?").any(axis=1)]
    filtered_dataset = filtered_dataset.astype(float)
    X = filtered_dataset[:, 0:-1]
    Y = filtered_dataset[:, -1]

    # Since all target > 0 is classified as having heart disease, we convert the value into 1
    Y[Y > 0] = 1

    Sw = within_class_scatter_matrix(X, Y)
    Sb = between_class_scatter_matrix(X, Y)
    X_cl_1 = X[Y == 1]
    X_cl_0 = X[Y == 0]

    # Get the projection vector
    w = projection_vector(within_cl_scatter=Sw, X_cl_1=X_cl_1, X_cl_0=X_cl_0)

    # Compute FLD before projection
    sep_before_projecton = fdr(
        projection_vector=w, within_cl_scatter=Sw, between_cl_scatter=Sb
    )

    print(f"Separability before projection: {sep_before_projecton:.3f}")

    # Project the data onto the discriminant
    X_projected = X @ w
    X_cl_1_projected = X_projected[Y == 1]
    X_cl_0_projected = X_projected[Y == 0]

    # Calculate separability on the projection
    sep_after_projection = fdr_1d(
        X_cl_1_projected=X_cl_1_projected, X_cl_0_projected=X_cl_0_projected
    )

    print(f"Separability after projection: {sep_after_projection:.3f}")



if __name__ == "__main__":
    main()

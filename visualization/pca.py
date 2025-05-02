import numpy as np
import matplotlib.pyplot as plt


def display_two_pca_projections(
    train_X,
    train_Y,
    pc_index_1,
    pc_index_2,
    pca_eigenvectors,
    explained_variance_ratio
):
    X_pca = train_X @ pca_eigenvectors

    plt.figure(figsize=(10, 7))

    # Plot each class with a different color
    for label in np.unique(train_Y):
        mask = train_Y == label
        plt.scatter(
            X_pca[mask, pc_index_1], X_pca[mask, pc_index_2], label=label, alpha=0.7
        )

    plt.xlabel(
        f"Principal Component 1 ({explained_variance_ratio[pc_index_1]:.2%} variance)"
    )
    plt.ylabel(
        f"Principal Component 2 ({explained_variance_ratio[pc_index_2]:.2%} variance)"
    )
    plt.title("PCA of Multiclass Dataset")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.show()

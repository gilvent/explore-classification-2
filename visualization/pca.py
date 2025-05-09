import numpy as np
import matplotlib.pyplot as plt


def display_two_pca_projections(
    train_X,
    train_Y,
    pc_index_1,
    pc_index_2,
    pca_eigenvectors,
    explained_variance_ratio,
    title="PCA Projection"
):
    X_pca = train_X @ pca_eigenvectors

    plt.figure(figsize=(8, 6))

    # Plot each class with a different color
    for label in np.unique(train_Y):
        mask = train_Y == label
        plt.scatter(
            X_pca[mask, pc_index_1], X_pca[mask, pc_index_2], label=label, alpha=0.7
        )

    plt.xlabel(
        f"Principal Component {pc_index_1 + 1} ({explained_variance_ratio[pc_index_1]:.2%} variance)"
    )
    plt.ylabel(
        f"Principal Component {pc_index_2 + 1} ({explained_variance_ratio[pc_index_2]:.2%} variance)"
    )
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)

    plt.show()

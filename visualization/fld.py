import numpy as np
import matplotlib.pyplot as plt


def display_1d_projection(
    X_projected: np.ndarray,
    Y: np.ndarray,
    title="1D Projection",
    label="Projected value",
):

    X_cl_1_projected = X_projected[Y == 1]
    X_cl_0_projected = X_projected[Y == 0]

    plt.figure(figsize=(6, 6))

    plt.scatter(
        X_cl_0_projected,
        np.zeros_like(X_cl_0_projected),
        c=(1, 0, 0, 0.5),
        marker="o",
        label="Class 0",
    )
    plt.scatter(
        X_cl_1_projected,
        np.zeros_like(X_cl_1_projected),
        c=(0, 0, 1, 0.3),
        marker="o",
        label="Class 1",
    )
    plt.axvline(x=0, color="black", linestyle="-", alpha=0.5)
    plt.xlabel(label)
    plt.yticks([])
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

import numpy as np
from dimensionality_reduction.class_separability_measures import (
    within_class_scatter_matrix,
    between_class_scatter_matrix,
)


class LinearDiscriminantAnalysis:

    def __init__(self, unique_classes):
        self.classes = unique_classes
        # Discriminants for projection
        self.W = np.array([])
        self.class_means = {}
        self.train_X_projected = np.array([])

    def train(self, train_X: np.ndarray, train_Y: np.ndarray):

        Sw = within_class_scatter_matrix(X=train_X, Y=train_Y)
        Sb = between_class_scatter_matrix(X=train_X, Y=train_Y)

        # Solve the eigenvalue problem
        eigvals, eigvecs = np.linalg.eig(np.linalg.inv(Sw) @ Sb)

        # Sort by descending eigenvalue
        sorted_indices = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, sorted_indices]
        eigvals = eigvals[sorted_indices]

        # Get (Num of classes - 1) Linear discriminants
        self.W = eigvecs[:, : (np.size(self.classes) - 1)]

        # Projection on LDA
        self.train_X_projected = train_X @ self.W

        # Compute class means in LDA space
        self.class_means = {
            cls: self.train_X_projected[train_Y == cls].mean(axis=0)
            for cls in self.classes
        }

    def predict(self, x):
        distances = {
            cls: np.linalg.norm(x - mean) for cls, mean in self.class_means.items()
        }

        return min(distances, key=lambda cl: distances[cl])

    def output(self, test_X):
        pred_Y = []
        test_X_projected = test_X @ self.W

        for x in test_X_projected:
            distances = {
                cls: np.linalg.norm(x - mean) for cls, mean in self.class_means.items()
            }

            # Pick closest class
            predicted_class = min(distances, key=lambda cl: distances[cl])
            pred_Y.append(predicted_class)

        return {
            "discriminants": self.W,
            "predictions": pred_Y,
        }

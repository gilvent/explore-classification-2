import numpy as np
from dimensionality_reduction.class_separability_measures import fld_vector


class FisherLinearDiscriminant:
    """
    Fisher Linear Discriminant.

    This classifier is specifically implemented for two-class cases.

    """

    def __init__(self, unique_classes):
        self.classes = unique_classes
        # Discriminants for projection
        self.W = np.array([])
        self.class_means = {}
        self.train_X_projected = np.array([])
        self.train_Y = None

    def fit(self, train_X: np.ndarray, train_Y: np.ndarray):
        self.train_Y = train_Y
        
        # Get the projection vector
        self.W = fld_vector(X=train_X, Y=train_Y)

        # Project the data onto the discriminant
        self.train_X_projected = train_X @ self.W

        # Compute class means in LDA space
        self.class_means = {
            cls: self.train_X_projected[train_Y == cls].mean(axis=0)
            for cls in self.classes
        }

    def transform(self, X):
        # Project into 1d
        return X @ self.W

    def output(self, test_X):
        pred_Y = []

        test_X_projected = test_X @ self.W

        # Use the middle point between the means
        X_1_projected_mean = np.mean(self.train_X_projected[self.train_Y == 1])
        X_0_projected_mean = np.mean(self.train_X_projected[self.train_Y == 0])
        threshold = (X_1_projected_mean + X_0_projected_mean) / 2

        for x in test_X_projected:
            # If larger than threshold, that it is the positive class
            pred_Y.append(1 if x >= threshold else 0)

        return {
            "discriminants": self.W,
            "predictions": pred_Y,
        }

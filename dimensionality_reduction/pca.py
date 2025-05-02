import numpy as np

class PrincipalComponentAnalysis:
    def __init__(self, n_components):
        self.n_components = n_components
        self.train_X_mean = None
        self.PC = None
        self.train_X = None
        self.eigenvalues = None

    def fit(self, train_X):
        # Subtract X with the mean
        self.train_X = train_X
        self.train_X_mean = np.mean(train_X, axis=0)
        train_X_centered = train_X - self.train_X_mean
        print(f"Data centered. Mean should be close to zero: {np.mean(train_X_centered, axis=0)}")

        # Calculate covariance matrix
        n_samples = train_X.shape[0]
        cov_matrix = np.dot(train_X_centered.T, train_X_centered) / (n_samples - 1)
        print(f"Covariance matrix shape: {cov_matrix.shape}")

        # Get eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort eigenvectors by eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[idx]
        self.eigenvectors = eigenvectors[:, idx]

        print("Sorted Eigenvalues", self.eigenvalues)

        # The eigenvalues represent the amount of variance explained by each principal component
        total_variance = np.sum(self.eigenvalues)
        self.explained_variance_ratio = self.eigenvalues / total_variance
        print("Explained variance ratio:", self.explained_variance_ratio)
        print(
            f"Selected components explain {np.sum(self.explained_variance_ratio[:self.n_components]):.2f} of variance"
        )

        self.PC = self.eigenvectors[:, :self.n_components]
            
        

    # Project train or test samples into l-dimensional subspace
    def transform(self, X):
        X_centered = X - self.train_X_mean
        return np.dot(X_centered, self.PC)
    

    def reconstruction_error(self):
        # Reconstruct data
        # This shows how the original data would look using selected components

        train_X_pca = self.transform(self.train_X)
        train_X_reconstructed = np.dot(train_X_pca, self.PC.T) + self.train_X_mean

        # Calculate reconstruction error
        reconstruction_error = np.mean(np.square(self.train_X - train_X_reconstructed))
        print(f"Reconstruction error: {reconstruction_error:.4f}")


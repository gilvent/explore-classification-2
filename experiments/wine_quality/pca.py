import numpy as np
from utils.data_preprocess import train_test_split
from dimensionality_reduction.pca import PrincipalComponentAnalysis
from visualization.pca import display_two_pca_projections


def main():
    dataset = np.genfromtxt(
        fname="data/winequality_white.csv", delimiter=";", dtype=str
    )
    # Remove header row
    dataset = dataset[1:]
    dataset = dataset.astype(float)
    X = dataset[:, 0:-1]
    Y = dataset[:, -1]

    train_X, train_Y, test_X, test_Y = train_test_split(
        X=X, Y=Y, test_split_ratio=0.3, shuffle=True, seed=13
    )

    # PCA
    pca = PrincipalComponentAnalysis(n_components=3)
    pca.fit(train_X=train_X)

    # This visualization is for top 2 principal components
    # Modify pc_index to check the other principal components
    display_two_pca_projections(
        train_X=train_X,
        train_Y=train_Y,
        pc_index_1=0,
        pc_index_2=1,
        pca_eigenvectors=pca.eigenvectors,
        explained_variance_ratio=pca.explained_variance_ratio,
    )


if __name__ == "__main__":
    main()

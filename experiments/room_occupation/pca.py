import numpy as np
from utils.data_preprocess import (
    to_seconds_since_midnight,
)
from dimensionality_reduction.pca import PrincipalComponentAnalysis
from visualization.pca import display_two_pca_projections


def preprocess(dataset):
    # Convert the second column (date time string) into numerical value
    X_2nd_col = dataset[:, 1]
    updated_X_2nd_col = []

    for datestr in X_2nd_col:
        time_in_seconds = to_seconds_since_midnight(datestr.replace('"', ""))
        updated_X_2nd_col.append(time_in_seconds)

    updated_X_2nd_col = np.array(updated_X_2nd_col)

    # Combine the updated second column with the rest, we don't use the first column
    X_rest = dataset[:, 2:-1].astype(float)
    X = np.hstack((updated_X_2nd_col.reshape(-1, 1), X_rest))
    Y = dataset[:, -1].astype(float)

    return (X, Y)


def main():
    training_set = np.loadtxt(
        fname="data/room_occupancy_datatraining.txt", delimiter=",", dtype=str
    )
    test_set = np.loadtxt(
        fname="data/room_occupancy_datatest.txt", delimiter=",", dtype=str
    )

    train_X, train_Y = preprocess(training_set)

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
        title="PCA Projection - Room Occupancy Dataset",
    )


if __name__ == "__main__":
    main()

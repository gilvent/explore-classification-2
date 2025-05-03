import numpy as np
from dimensionality_reduction.class_separability_measures import (
    fld_vector,
    trace_ratio,
)
from evaluation.roc_curve import predictions_by_threshold, print_roc_curve
from evaluation.confusion_matrix import confusion_matrix
from utils.data_preprocess import train_test_split
from visualization.fld import display_1d_projection


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

    classes = np.unique(Y)

    train_X, train_Y, test_X, test_Y = train_test_split(
        X=X, Y=Y, test_split_ratio=0.3, seed=13
    )

    # Separability before projection
    initial_class_separability = trace_ratio(X=train_X, Y=train_Y)

    # Get the projection vector
    w = fld_vector(X=train_X, Y=train_Y)

    # Project the data onto the discriminant
    X_projected = train_X @ w

    # Separability after projection
    projection_class_separability = trace_ratio(X=X_projected, Y=train_Y)

    display_1d_projection(
        X_projected=X_projected,
        Y=train_Y,
        title=f"Class separability, initial: {initial_class_separability:.3f}, projection: {projection_class_separability:.3f}",
    )

    # ROC Curve
    tpr = []
    fpr = []

    test_X_projected = test_X @ w

    # Get range of thresholds for classification
    lowest_X_projected = np.min(X_projected)
    highest_X_projected = np.max(X_projected)

    # Create 10 thresholds for ROC curve
    step = (highest_X_projected - lowest_X_projected) / 10
    thresholds = np.arange(lowest_X_projected, highest_X_projected + step + 0.1, step)

    for t in thresholds:
        thresholded_pred = predictions_by_threshold(
            discriminants=test_X_projected, threshold=t
        )
        conf_matrix = confusion_matrix(
            classes=classes, actual_Y=test_Y, pred_Y=thresholded_pred
        )

        tp = conf_matrix[1][1]
        fp = conf_matrix[0][1]
        tn = conf_matrix[0][0]
        fn = conf_matrix[1][0]

        tpr.append(tp / (tp + fn))
        fpr.append(fp / (fp + tn))

    print_roc_curve(fpr, tpr)


if __name__ == "__main__":
    main()

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
    display_1d_projection,
)
from evaluation.roc_curve import predictions_by_threshold, print_roc_curve
from evaluation.confusion_matrix import confusion_matrix
from utils.data_preprocess import train_test_split


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

    train_X, train_Y, test_X, test_Y = train_test_split(X=X, Y=Y, test_split_ratio=0.3)

    # Compute scatter matrices
    Sw = within_class_scatter_matrix(X=train_X, Y=train_Y)
    Sb = between_class_scatter_matrix(X=train_X, Y=train_Y)
    X_cl_1 = train_X[train_Y == 1]
    X_cl_0 = train_X[train_Y == 0]

    # Get the projection vector
    w = projection_vector(within_cl_scatter=Sw, X_cl_1=X_cl_1, X_cl_0=X_cl_0)

    # Compute FLD before projection
    sep_before_projection = fdr(
        projection_vector=w, within_cl_scatter=Sw, between_cl_scatter=Sb
    )

    fdr_before_proj_text = f"FDR (before projection): {sep_before_projection:.3f}"

    # Project the data onto the discriminant
    X_projected = train_X @ w
    X_cl_1_projected = X_projected[train_Y == 1]
    X_cl_0_projected = X_projected[train_Y == 0]

    # Calculate separability on the projection
    sep_after_projection = fdr_1d(
        X_cl_1_projected=X_cl_1_projected, X_cl_0_projected=X_cl_0_projected
    )

    fdr_after_proj_text = f"FDR (after projection): {sep_after_projection:.3f}"

    display_1d_projection(
        X_cl_0_projected=X_cl_0_projected,
        X_cl_1_projected=X_cl_1_projected,
        title=f"{fdr_before_proj_text}, {fdr_after_proj_text}",
    )

    # Get range of thresholds for classification
    lowest_X_projected = np.min(X_projected)
    highest_X_projected = np.max(X_projected)

    # Create 10 thresholds for ROC curve
    step = (highest_X_projected - lowest_X_projected) / 10
    thresholds = np.arange(lowest_X_projected, highest_X_projected + step + 0.1, step)

    # ROC Curve
    tpr = []
    fpr = []

    test_X_projected = test_X @ w
    classes = np.unique(Y)

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

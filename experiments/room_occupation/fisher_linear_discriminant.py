import numpy as np
from evaluation.metrics import accuracy_score, recall_score, precision_score, f1_score
from evaluation.confusion_matrix import confusion_matrix, display_confusion_matrix
from utils.data_preprocess import (
    to_seconds_since_midnight,
)
from evaluation.roc_curve import print_roc_curve, predictions_by_threshold
from dimensionality_reduction.class_separability_measures import (
    trace_ratio,
)
from visualization.fld import display_1d_projection
from classifiers.fisher_linear_discriminant import FisherLinearDiscriminant


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
    test_X, test_Y = preprocess(test_set)

    classes = np.unique(train_Y)

    # Separability before projection
    initial_class_separability = trace_ratio(X=train_X, Y=train_Y)

    # Fit to the model
    fld = FisherLinearDiscriminant(unique_classes=classes)
    fld.fit(train_X=train_X, train_Y=train_Y)

    # Separability after projection
    projection_class_separability = trace_ratio(X=fld.train_X_projected, Y=train_Y)

    display_1d_projection(
        X_projected=fld.train_X_projected,
        Y=train_Y,
        title="Room Occupancy/ 1D projection using FLD",
        label=f"Class separability, initial: {initial_class_separability:.3f}, projection: {projection_class_separability:.3f}",
    )

    # Make predictions on test data
    output = fld.output(test_X=test_X)
    pred_Y = output["predictions"]

    # Confusion matrix
    conf_matrix = confusion_matrix(classes=classes, actual_Y=test_Y, pred_Y=pred_Y)
    accuracy = accuracy_score(actual_Y=test_Y, pred_Y=pred_Y)
    recall = recall_score(tp=conf_matrix[1][1], fn=conf_matrix[1][0])
    precision = precision_score(tp=conf_matrix[1][1], fp=conf_matrix[0][1])
    f1 = f1_score(recall=recall, precision=precision)
    info_text = f"Accuracy: {accuracy:.2f}, F1 Score: {f1:.2f}"

    display_confusion_matrix(
        conf_matrix=conf_matrix,
        classes=classes,
        title="Room occupancy/Fisher Linear Discriminant",
        info=info_text,
    )

    # ROC Curve
    tpr = []
    fpr = []

    # Create 10 thresholds from lowest projected value to highest projected value
    lowest_X_projected = np.min(fld.train_X_projected)
    highest_X_projected = np.max(fld.train_X_projected)

    step = (highest_X_projected - lowest_X_projected) / 10
    thresholds = np.arange(lowest_X_projected, highest_X_projected + step + 0.1, step)

    test_X_projected = fld.transform(X=test_X)
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

    print_roc_curve(fpr, tpr, title="Room Occupancy/ROC Curve using FLD")


if __name__ == "__main__":
    main()

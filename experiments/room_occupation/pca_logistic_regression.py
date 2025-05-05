import numpy as np
from classifiers.logistic_regression import LogisticRegression
from evaluation.metrics import accuracy_score, recall_score, precision_score, f1_score
from evaluation.confusion_matrix import confusion_matrix, display_confusion_matrix
from utils.data_preprocess import (
    to_seconds_since_midnight,
)
from evaluation.roc_curve import print_roc_curve, predictions_by_threshold
from dimensionality_reduction.pca import PrincipalComponentAnalysis


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

    # Apply PCA after selecting top n-components that retains as much variance as possible
    pca = PrincipalComponentAnalysis(n_components=3)
    pca.fit(train_X=train_X)

    # Project the data
    train_X_pca = pca.transform(X=train_X)

    # Initialize the model
    classes = np.unique(train_Y)

    weights = np.asarray([0 for _ in range(0, train_X_pca.shape[1])])
    model = LogisticRegression(weights=weights)
    model.train(train_X=train_X_pca, train_Y=train_Y, iterations=1000, print_losses=True)

    # Project test data to make predictions
    test_X_pca = pca.transform(X=test_X)

    # Output the result
    output = model.output(threshold=0.5, test_X=test_X_pca)
    pred_probabilities = output["pred_probabilities"]
    pred_Y = output["pred_result"]

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
        title="Room occupancy/Logistic Regression w/ PCA",
        info=info_text,
    )

    # ROC Curve
    # We use posterior probabilities for affirmative class (where heart disease classified as PRESENT)
    tpr = []
    fpr = []

    for t in np.arange(0, 1.1, 0.05):
        thresholded_pred = predictions_by_threshold(
            discriminants=pred_probabilities, threshold=t
        )
        conf_matrix = confusion_matrix(
            classes=classes, actual_Y=test_Y, pred_Y=thresholded_pred
        )
        # print(conf_matrix)
        tp = conf_matrix[1][1]
        fp = conf_matrix[0][1]
        tn = conf_matrix[0][0]
        fn = conf_matrix[1][0]

        tpr.append(tp / (tp + fn))
        fpr.append(fp / (fp + tn))

    print_roc_curve(fpr, tpr)


if __name__ == "__main__":
    main()

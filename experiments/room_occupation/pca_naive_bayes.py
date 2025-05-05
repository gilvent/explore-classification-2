import numpy as np
from classifiers.naive_bayes import NaiveBayes
from evaluation.metrics import accuracy_score, recall_score, precision_score, f1_score
from evaluation.confusion_matrix import confusion_matrix, display_confusion_matrix
from utils.data_preprocess import (
    to_seconds_since_midnight,
)
from utils.enums import FeatureType
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
    test_set = np.loadtxt(fname="data/room_occupancy_datatest.txt", delimiter=",", dtype=str)

    train_X, train_Y = preprocess(training_set)
    test_X, test_Y = preprocess(test_set)

    # Initialize the model
    classes = np.unique(train_Y)

    # PCA
    pca = PrincipalComponentAnalysis(n_components=3)
    pca.fit(train_X=train_X)

    # Project the data
    train_X_pca = pca.transform(X=train_X)

    feature_types = [
        FeatureType.NUMERICAL.name,  # date time converted into numerical values based on time
        FeatureType.NUMERICAL.name,  # temperature
        FeatureType.NUMERICAL.name,  # humidty, continuous
        FeatureType.NUMERICAL.name,  # light
        FeatureType.NUMERICAL.name,  # CO2, continuous
        FeatureType.NUMERICAL.name,  # humidity ratio
    ]
    classifier = NaiveBayes(unique_classes=classes, feature_types=feature_types)

    classifier.train(train_X=train_X_pca, train_Y=train_Y)

    # Project test data to make predictions
    test_X_pca = pca.transform(X=test_X)

    output = classifier.output(test_X=test_X_pca)
    pred_Y = output["predictions"]

    # Confusion Matrix
    conf_matrix = confusion_matrix(classes=classes, actual_Y=test_Y, pred_Y=pred_Y)
    accuracy = accuracy_score(actual_Y=test_Y, pred_Y=pred_Y)
    recall = recall_score(tp=conf_matrix[1][1], fn=conf_matrix[1][0])
    precision = precision_score(tp=conf_matrix[1][1], fp=conf_matrix[0][1])
    f1 = f1_score(recall=recall, precision=precision)
    info_text = f"Accuracy: {accuracy:.2f}, F1 Score: {f1:.2f}"

    display_confusion_matrix(
        conf_matrix=conf_matrix,
        classes=classes,
        title="Room Occupation/Naive Bayes with PCA",
        info=info_text
    )

    # ROC Curve
    posteriors = np.array(output["posteriors"])
    affirmative_class_posteriors = posteriors[:, 1]
    tpr = []
    fpr = []

    for t in np.arange(0, 1.05, 0.05):
        thresholded_pred = predictions_by_threshold(
            discriminants=affirmative_class_posteriors, threshold=t
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

    print_roc_curve(fpr, tpr, title="Room Occupancy/ ROC Curve - Naive Bayes with PCA")


if __name__ == "__main__":
    main()

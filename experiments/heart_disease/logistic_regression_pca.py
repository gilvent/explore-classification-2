import numpy as np
from classifiers.logistic_regression import LogisticRegression
from evaluation.metrics import accuracy_score, recall_score, precision_score, f1_score
from evaluation.confusion_matrix import confusion_matrix, display_confusion_matrix
from utils.data_preprocess import train_test_split
from evaluation.roc_curve import print_roc_curve, predictions_by_threshold
from dimensionality_reduction.pca import PrincipalComponentAnalysis
from visualization.pca import display_two_pca_projections


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

    # Initialize the model
    classes = np.unique(Y)
    train_X, train_Y, test_X, test_Y = train_test_split(
        X=X, Y=Y, test_split_ratio=0.3, seed=13
    )

    pca = PrincipalComponentAnalysis(n_components=2)
    pca.fit(train_X=train_X)


    display_two_pca_projections(
        train_X=train_X,
        train_Y=train_Y,
        pc_index_1=0,
        pc_index_2=3,
        pca_eigenvectors=pca.eigenvectors,
        explained_variance_ratio=pca.explained_variance_ratio
    )

    train_X_pca = pca.transform(X=train_X)

    weights = np.asarray([0 for _ in range(0, train_X_pca.shape[1])])
    model = LogisticRegression(weights=weights)

    # Train on reduced features
    model.train(train_X=train_X_pca, train_Y=train_Y, iterations=3000)


    # Project the test data
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
        title="Heart Disease Dataset/Logistic Regression w/ PCA",
        info=info_text,
    )

    # ROC Curve
    # We use posterior probabilities for affirmative class (where heart disease classified as PRESENT)
    tpr = []
    fpr = []

    for t in np.arange(0, 1.1, 0.05):
        thresholded_pred = predictions_by_threshold(
            probabilities=pred_probabilities, threshold=t
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

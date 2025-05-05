import numpy as np
from classifiers.naive_bayes import NaiveBayes
from evaluation.metrics import accuracy_score, macro_f1_score
from evaluation.confusion_matrix import confusion_matrix, display_confusion_matrix
from utils.data_preprocess import train_test_split
from utils.enums import FeatureType
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

    # Project the data
    train_X_pca = pca.transform(X=train_X)

    # Initialize the classifier
    classes = np.unique(Y)
    feature_types = [
        FeatureType.NUMERICAL.name,  # fixed acidity
        FeatureType.NUMERICAL.name,  # volatile acidity
        FeatureType.NUMERICAL.name,  # citric acid
        FeatureType.NUMERICAL.name,  # residual sugar
        FeatureType.NUMERICAL.name,  # chlorides
        FeatureType.NUMERICAL.name,  # free sulfur dioxide
        FeatureType.NUMERICAL.name,  # total sulfur dioxide
        FeatureType.NUMERICAL.name,  # density
        FeatureType.NUMERICAL.name,  # pH
        FeatureType.NUMERICAL.name,  # sulphates
        FeatureType.NUMERICAL.name,  # alcohol
    ]
    classifier = NaiveBayes(unique_classes=classes, feature_types=feature_types)

    # Fit to the model
    classifier.train(train_X=train_X_pca, train_Y=train_Y)

    # Project the test data to make predictions
    test_X_pca = pca.transform(X=test_X)

    output = classifier.output(test_X_pca)
    pred_Y = output["predictions"]

    # Compute metrics and confusion matrix
    conf_matrix = confusion_matrix(classes=classes, actual_Y=test_Y, pred_Y=pred_Y)
    accuracy = accuracy_score(actual_Y=test_Y, pred_Y=pred_Y)
    macro_f1 = macro_f1_score(conf_matrix=np.array(conf_matrix))
    info_text = f"Accuracy: {accuracy:.2f}, Macro F1: {macro_f1:.2f}"

    display_confusion_matrix(
        conf_matrix=conf_matrix,
        classes=classes,
        title="White Wine Quality/Naive Bayes with PCA",
        info=info_text,
    )


if __name__ == "__main__":
    main()

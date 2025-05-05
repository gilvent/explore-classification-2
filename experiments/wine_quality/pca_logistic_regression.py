import numpy as np
from classifiers.multinominal_logistic_regression import MultinominalLogisticRegression
from evaluation.metrics import accuracy_score, macro_f1_score
from evaluation.confusion_matrix import confusion_matrix, display_confusion_matrix
from utils.data_preprocess import train_test_split
from dimensionality_reduction.pca import PrincipalComponentAnalysis


def main():
    dataset = np.genfromtxt(
        fname="data/winequality_white.csv", delimiter=";", dtype=str
    )
    # Remove header row
    dataset = dataset[1:]
    dataset = dataset.astype(float)
    X = dataset[:, 0:-1]
    Y = dataset[:, -1]

    classes = np.unique(Y)

    train_X, train_Y, test_X, test_Y = train_test_split(
        X=X, Y=Y, test_split_ratio=0.3, shuffle=True, seed=13
    )

    # Apply PCA after selecting top n-components that retains as much variance as possible
    pca = PrincipalComponentAnalysis(n_components=3)
    pca.fit(train_X=train_X)

    # Project the data
    train_X_pca = pca.transform(X=train_X)

    # Initialize the model
    weights = np.asarray([[0 for _ in classes] for _ in range(0, train_X_pca.shape[1])])
    bias = np.asarray([0.05 for _ in classes])
    model = MultinominalLogisticRegression(
        weights=weights, bias=bias, unique_classes=classes
    )

    # Train the model with PCA result
    model.train(
        train_X=train_X_pca, train_Y=train_Y, iterations=1000, print_losses=True
    )

    # Project the test data to make predictions
    test_X_pca = pca.transform(X=test_X)

    pred_probabilities = model.predict(test_X=test_X_pca)
    pred_Y = [classes[np.argmax(pbb)] for pbb in pred_probabilities]

    # Confusion Matrix
    accuracy = accuracy_score(actual_Y=test_Y, pred_Y=pred_Y)
    conf_matrix = confusion_matrix(classes=classes, actual_Y=test_Y, pred_Y=pred_Y)
    macro_f1 = macro_f1_score(conf_matrix=np.array(conf_matrix))
    info_text = f"Accuracy: {accuracy:.2f}, Macro F1: {macro_f1:.2f}"

    display_confusion_matrix(
        conf_matrix=conf_matrix,
        classes=classes,
        title="White Wine Quality/Logistic Regression with PCA",
        info=info_text,
    )


if __name__ == "__main__":
    main()

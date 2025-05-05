import numpy as np
from evaluation.confusion_matrix import confusion_matrix, display_confusion_matrix
from evaluation.metrics import accuracy_score, macro_f1_score
from utils.data_preprocess import train_test_split
from classifiers.linear_discriminant_analysis import LinearDiscriminantAnalysis
from dimensionality_reduction.class_separability_measures import generalized_eigenvalue


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

    # Class separability before projection
    initial_class_separability = generalized_eigenvalue(X=train_X, Y=train_Y)

    # Apply LDA
    model = LinearDiscriminantAnalysis(unique_classes=classes)
    model.train(train_X=train_X, train_Y=train_Y)

    # Class separability after projection
    projection_class_separability = generalized_eigenvalue(
        X=model.train_X_projected, Y=train_Y
    )
    class_separability_text = f"Class separability: {initial_class_separability:.3f} (Before LDA), {projection_class_separability:.3f} (After LDA)"
    print(class_separability_text)

    # Predictions on test data
    output = model.output(test_X=test_X)
    discriminants = output["discriminants"]
    print("LDA directions (in columns):\n", discriminants)

    pred_Y = output["predictions"]

    # Compute metrics
    accuracy = accuracy_score(actual_Y=test_Y, pred_Y=pred_Y)
    conf_matrix = confusion_matrix(classes=classes, actual_Y=test_Y, pred_Y=pred_Y)
    macro_f1 = macro_f1_score(conf_matrix=np.array(conf_matrix))
    info_text = f"Accuracy: {accuracy:.2f}, Macro F1: {macro_f1:.2f} \n{class_separability_text}"

    # Visualize confusion matrix
    display_confusion_matrix(
        conf_matrix=conf_matrix,
        classes=classes,
        title="White Wine Quality/Linear Discriminant Analysis",
        info=info_text,
    )


if __name__ == "__main__":
    main()

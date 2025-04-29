import matplotlib.pyplot as plt
import numpy as np


def predictions_by_threshold(probabilities: np.ndarray, threshold: float = 0.1):
    result = []
    for p in probabilities:
        result.append(1 if p >= threshold else 0)

    return result


# Approximated using Trapezoid Rule
def area_under_curve(sorted_fpr, sorted_tpr):
    total_area = 0

    for i in range(0, len(sorted_fpr) - 1):
        trapezoid_area = (
            (sorted_fpr[i + 1] - sorted_fpr[i])
            * (sorted_tpr[i] + sorted_tpr[i + 1])
            / 2
        )
        total_area += trapezoid_area

    return total_area


def print_roc_curve(fpr, tpr, title="ROC Curve"):
    sorted_indices = np.argsort(fpr)
    sorted_fpr = np.array(fpr)[sorted_indices]
    sorted_tpr = np.array(tpr)[sorted_indices]
    auc = area_under_curve(sorted_fpr=sorted_fpr, sorted_tpr=sorted_tpr)

    plt.title(title)
    plt.plot(sorted_fpr, sorted_tpr, "b", label="AUC = %0.2f" % auc)
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.show()

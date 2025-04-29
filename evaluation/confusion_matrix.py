import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def confusion_matrix(classes, actual_Y, pred_Y):
    counts = {}
    conf_matrix = []

    for actual in classes:
        for pred in classes:
            counts[(actual, pred)] = 0

    for actual, pred in zip(actual_Y, pred_Y):
        counts[(actual, pred)] += 1

    for row, actual_val in enumerate(classes):
        conf_matrix.append([])
        for pred_val in classes:
            conf_matrix[row].append(counts[(actual_val, pred_val)])

    return conf_matrix

def print_confusion_matrix(conf_matrix, classes, template="%12s"):

    template = " ".join([template] * (len(classes) + 1))
    header_labels = ["Predicted"] + [str(cl) for cl in classes]
    header = template % tuple(header_labels)
    secondary_header_labels = ["Actual"] + [""] * len(classes)
    secondary_header = template % tuple(secondary_header_labels)

    print("Confusion Matrix:")
    print(header)
    print(secondary_header)

    for row, counts in enumerate(conf_matrix):
        row_header_label = str(classes[row])
        labels = [row_header_label] + [str(val) for val in counts]
        row = template % tuple(labels)
        print(row)

def display_confusion_matrix(conf_matrix, classes, info=None, title="Confusion Matrix"):
    data = np.array(conf_matrix)
    fig, ax = plt.subplots()
    im = ax.imshow(
        X=data,
        cmap=mpl.colormaps["BuGn"],
    )

    ax.set_title(title, pad=20)
    plt.gca().xaxis.set_label_position("top")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")

    if info != None:
        plt.text(0.5, -0.1, info, ha="center", va="center", transform=ax.transAxes)

    threshold = im.norm(data.max()) / 2.0
    textcolors = ("green", "white")

    ax.set_xticks(
        range(len(classes)),
        labels=classes,
        rotation=0,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_yticks(range(len(classes)), labels=classes)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    for i in range(len(classes)):
        for j in range(len(classes)):
            text = ax.text(
                j,
                i,
                data[i, j],
                ha="center",
                va="center",
                color=textcolors[int(im.norm(data[i, j]) > threshold)],
            )

    fig.tight_layout()
    plt.show()

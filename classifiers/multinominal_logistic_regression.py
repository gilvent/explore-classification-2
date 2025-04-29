import numpy as np


class MultinominalLogisticRegression:
    """
    Parameters.

    unique_classes: A vector containing unique classes.
    
    weights: An (f, cl) matrix, where f = number of feature dimension, cl = number of classes
        The order of column (cl) should respect the order of unique classes.
        The order of row (f) should respect the order of features (columns of train_X passed to train() method)

    bias: A 1d vector of bias for each class
        The class order should respect unique_classes.
    """

    def __init__(
        self,
        weights: np.ndarray,
        bias: np.ndarray,
        unique_classes: np.ndarray,
        learning_rate=0.1,
    ):
        self.classes = unique_classes
        self.weights = weights
        self.bias = bias
        self.learning_rate = learning_rate
        self.loss_records = []

    def __scores(self, features_vector):
        return np.dot(np.transpose(self.weights), features_vector) + self.bias

    def __softmax(self, scores):
        # Shift values to 0 to avoid exponentiating on big numbers
        # https://stackoverflow.com/questions/54880369/implementation-of-softmax-function-returns-nan-for-high-inputs
        shifted_scores = scores - np.max(scores)
        exps = np.exp(shifted_scores)
        total_exps = np.sum(exps)

        return [(exp / total_exps) for exp in exps]

    def __cross_entropy_loss(self, actual_Y: np.ndarray, softmax_X: np.ndarray):
        epsilon = 1e-15
        clipped_softmax_X = [
            np.clip(class_scores, epsilon, 1 - epsilon) for class_scores in softmax_X
        ]
        total_loss = 0

        for index, y in enumerate(actual_Y):
            correct_class_index = np.where(self.classes == y)
            total_loss += -1 * np.log(clipped_softmax_X[index][correct_class_index])

        return total_loss / len(clipped_softmax_X)

    def train(self, train_X: np.ndarray, train_Y: np.ndarray, iterations=5, print_losses=True):

        for index in range(0, iterations):
            scores = np.asarray([self.__scores(features_vector=f) for f in train_X])
            self.softmax_X = np.asarray([self.__softmax(scores=s) for s in scores])
            avg_cross_entropy_loss = self.__cross_entropy_loss(
                actual_Y=train_Y, softmax_X=self.softmax_X
            )
            self.loss_records.append(avg_cross_entropy_loss)

            if (print_losses == True):
                print("Iteration", index, "Loss: ", avg_cross_entropy_loss)

            # Iterative optimization

            # True labels: (n, cl) matrix, where n = samples count, cl = classes count
            # We assign 1 for the class if it is the truth, 0 otherwise
            # Used to calculate how far the softmax is from the truth
            true_labels_Y = [
                [1 if cl == y else 0 for cl in self.classes] for y in train_Y
            ]

            softmax_diff = self.softmax_X - true_labels_Y
            weights_gradients = np.dot(np.transpose(train_X), softmax_diff)
            bias_gradient = np.sum(softmax_diff, axis=0)

            # Update weights and bias
            self.weights = (
                self.weights - self.learning_rate / train_X.shape[0] * weights_gradients
            )
            self.bias = (
                self.bias - self.learning_rate / train_X.shape[0] * bias_gradient
            )

        # Compute Loss for latest update
        scores = np.asarray([self.__scores(features_vector=f) for f in train_X])
        self.softmax_X = np.asarray([self.__softmax(scores=s) for s in scores])
        avg_cross_entropy_loss = self.__cross_entropy_loss(
            actual_Y=train_Y, softmax_X=self.softmax_X
        )
        self.loss_records.append(avg_cross_entropy_loss)

        print("Final Loss: ", avg_cross_entropy_loss)
        print("Completed", iterations, "iterations of training")

    # Returns the softmax_X
    def predict(self, test_X):
        return np.asarray(
            [self.__scores(features_vector=f) for f in test_X]
        )

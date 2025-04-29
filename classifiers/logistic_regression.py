import numpy as np


class LogisticRegression:

    def __init__(self, weights, learning_rate=0.1):
        self.weights = weights
        self.bias = 0.05
        self.learning_rate = learning_rate
        self.loss_records = []

    def __score(self, features_vector):
        return (
            np.sum([x * w for x, w in zip(features_vector, self.weights)]) + self.bias
        )

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-1 * z))

    def __cross_entropy_loss(self, actual_Y: np.ndarray, pred_Y_pbb: np.ndarray):
        # Current calculation is only for binary class,
        # where pred_y_pbb is probability of true class and (1 - pred_y_pbb) is of false class
        epsilon = 1e-15
        pred_pbb = np.clip(pred_Y_pbb, epsilon, 1 - epsilon)

        log_likelihoods = np.asarray(
            [
                y * np.log(y_pbb) + (1 - y) * np.log(1 - y_pbb)
                for y, y_pbb in zip(actual_Y, pred_pbb)
            ]
        )
        total_likelihoods = np.sum(log_likelihoods)
        avg_cross_entropy_loss = -1 * total_likelihoods / pred_Y_pbb.shape[0]
        return avg_cross_entropy_loss

    def train(
        self, train_X: np.ndarray, train_Y: np.ndarray, iterations=5, print_losses=False
    ):
        for index in range(0, iterations):
            scores = [self.__score(features_vector=f) for f in train_X]
            self.predicted_probabilities = np.asarray(
                [self.__sigmoid(z) for z in scores]
            )

            avg_cross_entropy_loss = self.__cross_entropy_loss(
                actual_Y=train_Y, pred_Y_pbb=self.predicted_probabilities
            )
            self.loss_records.append(avg_cross_entropy_loss)

            if print_losses == True:
                print("Iteration", index, "Loss: ", avg_cross_entropy_loss)

            # Iterative optimization

            # Calculate the gradient of weights
            error_vector = self.predicted_probabilities - train_Y
            n = np.shape(train_X)[0]
            weight_gradients = 1 / n * np.dot(np.transpose(train_X), error_vector)

            # Calculate gradient of bias
            bias_gradient = 1 / n * np.sum(error_vector)

            # Update weights and bias
            self.weights = self.weights - self.learning_rate * weight_gradients
            self.bias = self.bias - self.learning_rate * bias_gradient

        # Compute Loss for latest update
        scores = [self.__score(features_vector=f) for f in train_X]
        self.predicted_probabilities = np.asarray([self.__sigmoid(z) for z in scores])
        avg_cross_entropy_loss = self.__cross_entropy_loss(
            actual_Y=train_Y, pred_Y_pbb=self.predicted_probabilities
        )
        self.loss_records.append(avg_cross_entropy_loss)

        print("Final Loss: ", avg_cross_entropy_loss)
        print("Completed", iterations, "iterations of training")

    def output(self, test_X, threshold=0.5):
        scores = [self.__score(features_vector=f) for f in test_X]
        probabilities = [self.__sigmoid(z) for z in scores]
        return {
            "discriminants": scores,
            "pred_probabilities": probabilities,
            "pred_result": [1 if pbb >= threshold else 0 for pbb in probabilities],
        }

import numpy as np
import math
from utils.enums import FeatureType


class NaiveBayes:

    def __init__(self, feature_types: list, unique_classes):
        self.feature_types = feature_types
        self.classes = unique_classes

    # Use 1-D Gaussian PDF formula
    def __gaussian_pdf(self, mean: float, std: float, x: float):
        coefficient = 1 / (math.sqrt(2 * math.pi) * std)
        # Prevent division by zero
        epsilon = 1e-9
        variance = math.pow(std, 2) + epsilon
        exponent = math.exp(-1 / (2 * variance) * math.pow(x - mean, 2))
        return coefficient * exponent

    def __probability_by_frequency(self, feature_index, cls, x):
        # If the key does not exist, then this feature has no occurence in this class during training.
        # However, we just assume a very low probability for this feature so that the entire probability is not nullified.
        if (feature_index, cls, x) not in self.feature_freq:
            return 0.00000000001

        return self.feature_freq[(feature_index, cls, x)] / self.class_freq[cls]

    def __likelihood(self, target_class, feature_index, x):
        if self.feature_types[feature_index] == FeatureType.NUMERICAL.name:
            mean, std = self.pdf_params[(feature_index, target_class)]
            return self.__gaussian_pdf(mean, std, x)

        if self.feature_types[feature_index] == FeatureType.CATEGORICAL.name:
            return self.__probability_by_frequency(feature_index, target_class, x)

        return 0

    def __inputs_likelihood(self, nd_features, target_class):
        likelihood_product = 1
        for feature_index, x in enumerate(nd_features):
            likelihood = self.__likelihood(
                target_class=target_class, feature_index=feature_index, x=x
            )
            likelihood_product *= likelihood

        return likelihood_product

    def train(self, train_X: np.ndarray, train_Y: np.ndarray):
        if len(train_X) <= 0:
            print("Training failed: Input data required")
            return False
        if len(train_Y) <= 0:
            print("Training failed: Output data required")
            return False

        self.pdf_params = {}
        self.feature_freq = {}
        self.prioris = {}
        self.class_freq = {cl: 0 for cl in self.classes}
        self.calculated_likelihoods = {}

        # Calculate parameters for probability distribution function
        for cl in self.classes:
            rows_with_cl_output = train_X[train_Y == cl]
            transposed_rows = np.transpose(rows_with_cl_output)

            for feature_index, values in enumerate(transposed_rows):

                # For numerical feature:
                # Store the mean and standard deviation in given class
                if self.feature_types[feature_index] == FeatureType.NUMERICAL.name:
                    self.pdf_params[(feature_index, cl)] = (
                        np.mean(values),
                        np.std(values),
                    )

                # For categorical feature:
                # Store the frequency of each category in given class
                if self.feature_types[feature_index] == FeatureType.CATEGORICAL.name:
                    for val in values:
                        if (feature_index, cl, val) in self.feature_freq:
                            self.feature_freq[(feature_index, cl, val)] += 1
                        else:
                            self.feature_freq[(feature_index, cl, val)] = 1

        # Count the classes frequency for priori probabilities P(Y)
        for cl in train_Y:
            self.class_freq[cl] += 1

        # Calculate priori probabilities for each class
        for cl in self.classes:
            self.prioris[cl] = self.class_freq[cl] / len(train_Y)

        return True

    def output(self, test_X: np.ndarray):
        output_data = {"predictions": [], "posteriors": []}

        for nd_features in test_X:
            posterior_by_class = []

            for cl in self.classes:
                # Likelihood of inputs in class
                likelihood_in_class = self.__inputs_likelihood(
                    nd_features=nd_features, target_class=cl
                )

                # Store p(x | w) * P(w) for each class
                posterior_by_class.append(likelihood_in_class * self.prioris[cl])

            # Scaling factor: Sum of (likelihood of inputs * Priori) for all classes
            scaling_factor = np.sum(posterior_by_class)

            # Compute the posterior probability for each class.
            # Posterior = Likelihood in class * Priori / Scaling factor
            posterior_by_class = posterior_by_class / scaling_factor

            output_data["posteriors"].append(posterior_by_class)

            predicted_class = self.classes[np.argmax(posterior_by_class)]
            output_data["predictions"].append(predicted_class)

        return output_data

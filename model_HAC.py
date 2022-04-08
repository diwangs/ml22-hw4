""" 
Keep model implementations in here.

This file is where you will write all of your code!
"""

import numpy as np

class Model(object):
    """ Abstract model object."""

    def __init__(self):
        raise NotImplementedError()

    def fit_predict(self, X):
        """ Predict.

        Args:
            X: A dense matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()



class AgglomerativeClustering(Model):

    def __init__(self, n_clusters = 2, linkage = 'single'):
        """
        Args:
            n_clusters: number of clusters
            linkage: linkage criterion
        """

        self.n_clusters = n_clusters
        self.linkage = linkage

    def fit_predict(self, X):
        """ Fit and predict.

        Args:
            X: A dense matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """

        # Initialize 
        y = np.arange(X.shape[0])

        while np.unique(y).shape[0] != self.n_clusters:
            # Get cluster indices
            y_unique = np.unique(y)
            
            # Find clusters that have minimal distance
            y_pairs = np.array(np.meshgrid(y_unique, y_unique)).T.reshape(-1, 2)
            min_distance = np.Inf # large number
            min_pair = []

            for y_pair in y_pairs:

                if y_pair[0] >= y_pair[1]:
                    # No need to examine identical cluster distance
                    continue

                # Get two clusters
                filter_a = [y_i == y_pair[0] for y_i in y]
                filter_b = [y_i == y_pair[1] for y_i in y]

                X_a = X[filter_a]
                X_b = X[filter_b]

                # Calculate distance
                distance = self.calc_distance(X_a, X_b)

                # TODO: tie-breaker?
                if distance < min_distance:
                    min_distance = distance
                    min_pair = y_pair

            # Merge clusters
            y = np.where(y == min_pair[1], min_pair[0], y)

        return y

    def calc_distance(self, X_a, X_b):
        if self.linkage == 'single':
            min_distance = np.Inf
            for x_a in X_a:
                for x_b in X_b:
                    distance = np.linalg.norm(x_a - x_b)
                    if distance < min_distance:
                        min_distance = distance
            return min_distance

        elif self.linkage == 'complete':
            max_distance = -np.Inf
            for x_a in X_a:
                for x_b in X_b:
                    distance = np.linalg.norm(x_a - x_b)
                    if distance > max_distance:
                        max_distance = distance
            return max_distance

        elif self.linkage == 'average':
            distance_sum = 0
            for x_a in X_a:
                for x_b in X_b:
                    distance_sum += np.linalg.norm(x_a - x_b)
            return distance_sum / (X_a.shape[0] * X_b.shape[0])

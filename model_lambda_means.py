""" 
Keep model implementations in here.

This file is where you will write all of your code!
"""

import numpy as np
from tqdm import tqdm
from scipy.sparse import linalg

class Model(object):
    """ Abstract model object.

    Contains a helper function which can help with some of our datasets.
    """

    def __init__(self, nfeatures):
        self.num_input_features = nfeatures


    def fit(self, *, X, iterations):
        """ Fit the model.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            iterations: int giving number of clustering iterations
        """
        raise NotImplementedError()


    def predict(self, X):
        """ Predict.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()


    def _fix_test_feats(self, X):
        """ Fixes some feature disparities between datasets.
        Call this before you perform inference to make sure your X features
        match your weights.
        """
        num_examples, num_input_features = X.shape
        if num_input_features < self.num_input_features:
            X = X.copy()
            X._shape = (num_examples, self.num_input_features)
        if num_input_features > self.num_input_features:
            X = X[:, :self.num_input_features]
        return X


class LambdaMeans(Model):

    def __init__(self, *, nfeatures, lambda0):
        super().__init__(nfeatures)
        """
        Args:
            nfeatures: size of feature space (only needed for _fix_test_feats)
            lambda0: A float giving the default value for lambda
        """
        # TODO: Initializations etc. go here.
        self.nfeatures = nfeatures
        self.lambda0 = lambda0

        self.centroids = np.zeros((1, self.nfeatures))

    def fit(self, *, X, iterations):
        """
        Fit the LambdaMeans model.
        Note: labels are not used here.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            iterations: int giving number of clustering iterations
        """
        # Initialize one cluster as the mean of all X
        self.centroids[0] = np.mean(X, axis=0)

        # Initialize lambda to average if lambda == 0
        if self.lambda0 == 0:
            distances = [self.calc_distance(x, self.centroids[0]) for x in X]
            self.lambda0 = np.mean(np.array(distances))
            print("No lambda, is now", self.lambda0)

        # Train
        for _ in range(iterations):
            # Initialize cluster?
            y = np.zeros(X.shape[0])
            
            # E-step
            for i, x in tqdm(enumerate(X)):
                distances = [self.calc_distance(x.toarray(), c) for c in self.centroids]
                min_distance, min_idx = min((val, idx) for (idx, val) in enumerate(distances))

                if min_distance > self.lambda0:
                    # create new centroid
                    self.centroids = np.append(self.centroids, x.toarray(), axis=0)
                    y[i] = self.centroids.shape[0] - 1
                else:
                    y[i] = min_idx

            # M-step
            for i in range(self.centroids.shape[0]):
                filter0 = [y_i == i for y_i in y]
                cluster = X.toarray()[filter0]
                if cluster.shape[0] != 1:
                    print(cluster.shape)
                if cluster.shape[0] == 0:
                    self.centroids[i] = np.zeros(self.nfeatures)
                else:
                    self.centroids[i] = np.mean(cluster, axis=0)

            print(self.centroids.shape)


    def predict(self, X):
        """ Predict.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        y = []
        for x in tqdm(X):
            distances = [self.calc_distance(x, c) for c in self.centroids]
            min_distance, min_idx = min((val, idx) for (idx, val) in enumerate(distances))
            y.append(min_idx)
        
        return y

    def calc_distance(self, x_a, x_b):
        return np.linalg.norm(x_a - x_b)

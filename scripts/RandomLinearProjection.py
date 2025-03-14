import numpy as np
from sklearn.base import OneToOneFeatureMixin, TransformerMixin, BaseEstimator

class RandomLinearProjection(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    def __init__(self, dim, random_state=0):
        self.target_dim = dim
        self.rng = np.random.default_rng(random_state)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # random linear projection
        # gen random matric
        original_dim = X.shape[1]
        random_matrix = self.rng.random(size=(original_dim, self.target_dim))
        # do projection!
        X = np.dot(X, random_matrix)
        return X

    def _more_tags(self):
        return {"stateless": True}

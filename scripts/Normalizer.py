import numpy as np
from sklearn.base import OneToOneFeatureMixin, TransformerMixin, BaseEstimator

class Normalizer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return Normalizer.normalize(X)

    def _more_tags(self):
        return {"stateless": True}

    @staticmethod
    def normalize(X):
        row_sums = X.sum(axis=1)
        X = X / row_sums[:, np.newaxis]
        return X

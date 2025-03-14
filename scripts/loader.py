import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

sys.path.append(str(Path(__file__).parent.parent))

class Loader:
    def __init__(self, infile, addone=False, scale=1, use_sliding=False, n_threads=1, check_last=True):
        self.infile = infile
        self.scale = scale
        self.use_sliding = use_sliding
        self.n_threads = n_threads
        self.check_last = check_last
        self.addone = addone

    def load(self):
        X = self.bbvdict2vec(self.infile, self.addone, self.scale, self.n_threads, self.check_last)
        if self.use_sliding:
            X = Loader.sliding(X)
        return X

    @staticmethod
    def bbvdict2vec_actor(lines):
        counter_dict = {}
        for line in lines:
            splitted = line.strip().split(':')
            idxs = [int(x) for x in splitted[1::2]]
            vals = [int(x) for x in splitted[2::2]]
            for idx, val in zip(idxs, vals):
                counter_dict[idx] = counter_dict.get(idx, 0) + val
        return counter_dict

    @staticmethod
    def bbvdict2vec(infile, addone, scale, n_threads, check_last=True):
        import gzip
        X = []
        with gzip.open(infile, 'rt') as f:
            from functools import reduce
            groups = Loader.batched((line for line in f if line.startswith("T:")), n=scale)
            drop_last = len(groups[-1]) < scale
            X = Parallel(n_jobs=n_threads, prefer="threads")(
                delayed(Loader.bbvdict2vec_actor)(group) for group in groups
            )

            counter_keys = reduce(lambda x, y: x.union(y.keys()), X, set())
            allkey_dict = dict.fromkeys(range(1, max(counter_keys) + 1), 0)
            X.append(allkey_dict)

        from sklearn.feature_extraction import DictVectorizer
        v = DictVectorizer(sparse=True, sort=True, dtype=np.float32)
        X = v.fit_transform(X)
        X = X.toarray()
        X = X[:-2] if check_last and drop_last else X[:-1]
        # to avoid divide by zero in js
        if addone:
            X = X + 1
        return X

    @staticmethod
    def batched(iterable, n):
        import itertools
        # batched('ABCDEFG', 3) --> ABC DEF G
        if n < 1:
            raise ValueError('n must be at least one')
        it = iter(iterable)
        batches = []
        while batch := tuple(itertools.islice(it, n)):
            batches.append(batch)
        return batches

    @staticmethod
    def sliding(X):
        X_past = np.roll(X, 1, axis=0)
        X_past[0] = X[0]
        X_sliding = np.concatenate((X_past, X), axis=1)
        return X_sliding


"""Get some clustering data for analysis"""
import collections
import datetime
import itertools
import time

import numpy as np
# This must be our fork of pySAP
from pysapc import SAP
from scipy.sparse import coo_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import cosine_similarity


def make_affinity_matrix(phenotypes):
    """
    Generates an affinity matrix from a list of phenotypes as the squared
    cosine similarity of all common channels.
    """
    first, *rest = phenotypes
    idx = first.index
    for df in rest:
        idx = idx.intersection(df.index)

    vals = [df.loc[idx].values for df in phenotypes]
    data = np.concatenate(vals, axis=1).T

    C = cosine_similarity(data)
    C = np.square(C)
    return C


def iter_clusters(matrix, labels):
    """Iterates tuples of (cluster submatrix, cluster indices)"""
    labelset = np.unique(labels)
    #assert (labelset == np.arange(len(labelset))).all(), labelset
    for n in labelset:
        mask = labels == n
        index = np.ix_(mask, mask)
        submat = matrix[index]
        yield submat, index


class PerfectClusterScorer:
    def __init__(self, n_model, n_phent):
        self.n_model = n_model
        self.n_phent = n_phent
        self.lo = np.arange(0, n_model*n_phent, n_phent, dtype=np.int)
        self.hi = self.lo + n_phent

    def is_perfect(self, cluster_indices):
        idx = np.sort(cluster_indices.ravel())
        if len(idx) != self.n_model:
            return False
        return np.all((self.lo <= idx) & (idx < self.hi))

    def __call__(self, matrix, labels, X=None, y=None):
        # Failure of clusterer to converge
        if np.all(labels == -1):
            return -1

        score = 0
        for (_, (_, idx)) in iter_clusters(matrix, labels):
            if self.is_perfect(idx):
                score += 1
        return score


class AffinityPropagationClusterer:
    def __init__(self, *, preference=None, convergence_iter=15,
                 max_iter=200, damping=0.5, copy=True, random_state=0):
        self.preference = preference
        self.convergence_iter = convergence_iter
        self.max_iter = max_iter
        self.damping = damping
        self.copy = copy
        self.random_state = random_state

    def fit(self, S, thresh=0.5):
        S = coo_matrix(S, copy=self.copy, dtype=np.float)

        # Allowing thresh to be None allows this whole conversion to happen
        # elsewhere
        if thresh is not None:
            S[S < thresh] = 0.0

        if self.random_state is not None:
            np.random.seed(self.random_state)

        sap = SAP(preference=self.preference,
                  convergence_iter=self.convergence_iter,
                  max_iter=self.max_iter,
                  damping=self.damping,
                  verboseIter=None,
                  parallel=True)

        tstart = time.time()
        centers, labels, n_iter = sap.fit_predict(S)
        wall_time = time.time() - tstart

        self.centers_ = centers
        self.labels_ = labels
        self.n_iter_ = n_iter
        self.wall_time_ = wall_time

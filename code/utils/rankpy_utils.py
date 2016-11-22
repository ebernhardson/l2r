import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import chain

from sklearn.model_selection import GroupKFold
from rankpy.queries import Queries
from rankpy.models import LambdaMART as LambdaMARTModel

class LambdaMART(object):
    def __init__(self, metric='NDCG', n_estimators=100, max_depth=None,
                max_leaf_nodes=7, max_features=None, min_samples_split=2,
                min_samples_leaf=1, shrinkage=0.1, use_newton_method=True,
                use_random_forest=0, random_thresholds=False, subsample=1.0,
                use_logit_boost=False, use_ada_boost=False, estopping=50,
                min_n_estimators=1, base_model=None, n_jobs=1, random_state=None):
        self.feature_names = None

        self.params = {
            'metric': metric,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'max_leaf_nodes': max_leaf_nodes,
            'max_features': max_features,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'shrinkage': shrinkage,
            'use_newton_method': use_newton_method,
            'use_random_forest': use_random_forest,
            'random_thresholds': random_thresholds,
            'subsample': subsample,
            'use_logit_boost': use_logit_boost,
            'use_ada_boost': use_ada_boost,
            'estopping': estopping,
            'min_n_estimators': min_n_estimators,
            'base_model': base_model,
            'n_jobs': n_jobs,
            'random_state': random_state,
        }


    def __str__(self):
        return self.__repr__()

    def __repr(self):
        return ("%s(metric='%s', n_estimators=%d, max_depth=%d, max_leaf_nodes=%d,\n"
                "max_features=%d, min_samples_split=%d, min_samples_leaf=%d,\n"
                "shrinkage=%f, use_newton_method=%s, use_random_forest=%d,\n"
                "random_thresholds=%s, subsample=%f, use_logit_boost=%s, use_ada_boost=%s,\n"
                "estopping=%d, min_n_estimators=%d, n_jobs=%d, random_state=%s,\n"
                "base_model=%s)" % (
                self.__class__.__name__,
                self.param["metric"],
                self.params["n_estimators"],
                self.params["max_depth"],
                self.params["max_leaf_nodes"],
                self.params["max_features"],
                self.params["min_samples_split"],
                self.params["min_samples_leaf"],
                self.params["shrinkage"],
                self.params["use_newton_method"],
                self.params["use_random_forest"],
                self.params["random_thresholds"],
                self.params["subsample"],
                self.params["use_logit_boost"],
                self.params["use_ada_boost"],
                self.params["estopping"],
                self.params["min_n_estimators"],
                self.params["n_jobs"],
                self.params["random_state"],
                str(self.params["base_model"]),
            ))

    def _build_query_indptr(self, ids):
        """
        The query index pointer into the feature_vectors and relevance_scores
        array, i.e. the document feature vectors,
        ``feature_vectors[query_indptr[i]:query_indptr[i + 1]]``, and the
        corresponding relevance scores,
        ``relevance_scores[query_indptr[i]:query_indptr[i + 1]]``,
        are the feature vectors and relevance scores for the i-th
        query documents.
        """
        query_indptr = [0]
        query_ids = []
        prev_qid = None
        for qid in ids:
            if qid == prev_qid:
                query_indptr[-1] += 1
            else:
                query_ids.append(qid)
                query_indptr.append(query_indptr[-1] + 1)
                prev_qid = qid
        return query_indptr, query_ids

    def _build_queries(self, X, y, ids, w):
        query_indptr, query_ids = self._build_query_indptr(ids)
        q = Queries(X, y, query_indptr, query_ids=query_ids)
        # weights as per query instead of per-row ... just guess
        wn = [np.mean(w[query_indptr[i]:query_indptr[i+1]]) for i in range(len(query_indptr)-1)]
        wn = [w[i] for i in query_indptr[:-1]]
        return q, np.ascontiguousarray(wn, dtype='float64')

    def fit(self, X, y, ids, weight=None, feature_names=None):
        self.feature_names = feature_names
        # Unfortunately rankpy only works with integer labels...
        # This is far from perfect, but works as a first try
        y = (np.asanyarray(y) * 5).astype(np.intc)
        #  Split out a 10% validation set
        splitter = GroupKFold(10)
        train, valid = next(splitter.split(X, None, ids))

        X_train, X_valid, y_train, y_valid, ids_train, ids_valid, w_train, w_valid = chain.from_iterable(
            ((a[train], a[valid]) for a in [X, y, ids, weight]))

        q_train, w_train = self._build_queries(X_train, y_train, ids_train, w_train)
        q_valid, w_valid = self._build_queries(X_valid, y_valid, ids_valid, w_valid)

        self.model = LambdaMARTModel(**self.params)
        self.model.fit(q_train, w_train, q_valid, w_valid)
        return self

    def predict(self, X, ids, weight, feature_names=None):
        self.feature_names = feature_names
        query_indptr, query_ids = self._build_query_indptr(ids)
        # We wont be using this, but Queries wont instantiate without it
        y = np.zeros(X.shape[0])
        q = Queries(X, y, query_indptr, query_ids=query_ids)
        y_pred = self.model.predict(q, n_jobs=self.params['n_jobs'])
        return y_pred

    def plot_importance(self):
        if self.feature_names is None:
            raise Exception('No feature names available')

        importance = self.model.feature_importances(self.params['n_jobs'])

        # stolen from xgboost
        tuples = zip(self.feature_names, importance)
        tuples = sorted(tuples, key=lambda x: x[1])
        labels, values = tuples

        self.save_topn_features(labels, values)

        _, ax = plt.subplots(1, 1)
        ylocs = np.arange(len(values))
        ax.barh(ylocs, values, align='center', height=0.2)
        for x, y in zip(values, yloc):
            ax.text(x + 1, y, x, va='center')
        ax.set_yticks(ylocs)
        ax.set_yticklabels(labels)

        xlim = (0, max(values) * 1.1)
        ax.set_xlim(xlim)

        ylim = (-1, len(importance))
        ax.set_ylim(ylim)

        ax.grid()
        return ax

    def save_topn_features(self, labels, values, fname="LambdaMART_topn_features.txt", topn=-1):
        if topn == -1:
            topn = len(labels)
        else:
            topn = min(topn, len(labels))
        with open(fname, "w") as f:
            for i in range(topn):
                f.write("%s = %f" % (labels[i], values[i]))

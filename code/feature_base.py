import numpy as np
import pandas as pd

import os

import config
from utils import np_utils, table_utils

# Base class for measuring correlation/similarity/distance
# between the search query and page information
class BaseEstimator(object):
    def __init__(self, obs_corpus, target_corpus, aggregation_mode, id_list=None):
        self.obs_corpus = obs_corpus
        self.N = len(obs_corpus)
        # for standalone features, not sure why a range though
        self.target_corpus = range(self.N) if target_corpus is None else target_corpus
        # id_list is used for group based relevance/distance detectors
        self.id_list = range(self.N) if id_list is None else id_list
        # aggregation for list features, e.g. intersect points
        self.aggregation_mode, self.aggregator = self._check_aggregation_mode(aggregation_mode)

    def _check_aggregation_mode(self, aggregation_mode):
        valid_aggregation_modes = ["", "size", "mean", "std", "max", "min", "median"]
        if isinstance(aggregation_mode, str):
            aggregation_mode = [aggregation_mode]
        if isinstance(aggregation_mode, list):
            for m in aggregation_mode:
                assert m.lower() in valid_aggregation_modes, "Wrong aggregation mode: %s" % (m)
            aggregation_mode = [m.lower() for m in aggregation_mode]

        aggregator = [None if m == "" else getattr(np, m) for m in aggregation_mode]

        return aggregation_mode, aggregator

    def transform(self):
        # generate scores
        score = list(map(self.transform_one, self.obs_corpus, self.target_corpus, self.id_list))
        # aggregation
        if isinstance(score[0], list):
            # single aggregation
            res = np.zeros((self.N, len(self.aggregator)), dtype=float)
            for m, aggregator in enumerate(self.aggregator):
                for i in range(self.N):
                    try:
                        s = aggregator(score[i])
                    except:
                        s = config.MISSING_VALUE_NUMERIC
                    res[i,m] = s
        else:
            res = np.asarray(score)
        return res


# Wrapper for generating standalone features, e.g.
# count of words in a search query
class StandaloneFeatureWrapper(object):
    def __init__(self, generator, dfAll, obs_fields, param_list, feat_dir, logger, deduplicate=False):
        self.generator = generator
        self.dfAll = dfAll
        self.obs_fields = obs_fields
        self.param_list = param_list
        self.feat_dir = feat_dir
        self.logger = logger
        self.deduplicate = deduplicate

    def go(self):
        y_train = self.dfAll["relevance"].values
        for obs_field in self.obs_fields:
            if obs_field not in self.dfAll.columns:
                self.logger.info("Skip %s" % (obs_field))
                continue
            if self.deduplicate == True:
                obs_corpus = self.dfAll[obs_field].drop_duplicates().values
            else:
                obs_corpus = self.dfAll[obs_field].values
            estimator = self.generator(obs_corpus, None, *self.param_list)
            x = estimator.transform()
            if self.deduplicate == True:
                # re-duplicate the values
                x_df = pd.DataFrame(zip(obs_corpus, x), columns=['src', 'est']).set_index(['src'])
                # We need a list to ensure we get back a DataFrame and not a Series
                x = self.dfAll[[obs_field]].join(x_df, on=[obs_field], how='left')['est'].values
                # This feels like a hack, but we have ended up with an ndarray of ndarray on
                # aggregations and need to fix it
                if type(x[0]) == np.ndarray:
                    x = np.vstack(x)
            if isinstance(estimator.__name__(), list):
                for i, feat_name in enumerate(estimator.__name__()):
                    dim = 1
                    fname = "%s_%s_%dD" % (feat_name, obs_field, dim)
                    table_utils._write(os.path.join(self.feat_dir, fname+config.FEAT_FILE_SUFFIX), x[:,i])
                    corr = np_utils._corr(x[:,i], y_train)
                    self.logger.info("%s (%dD): corr=%.6f" % (fname, dim, corr))
            else:
                dim = np_utils._dim(x)
                fname = "%s_%s_%dD" % (estimator.__name__(), obs_field, dim)
                table_utils._write(os.path.join(self.feat_dir, fname+config.FEAT_FILE_SUFFIX), x)
                if dim == 1:
                    corr = np_utils._corr(x, y_train)
                    self.logger.info("%s (%dD): corr=%.6f" % (fname, dim, corr))


# wrapper for generating pairwise feature, e.g.,
# intersect count of words between query and page title
class PairwiseFeatureWrapper(object):
    def __init__(self, generator, dfAll, obs_fields, target_fields, param_list, feat_dir, logger, deduplicate=False):
        self.generator = generator
        self.dfAll = dfAll
        self.obs_fields = obs_fields
        self.target_fields = target_fields
        self.param_list = param_list
        self.feat_dir = feat_dir
        self.logger = logger
        self.deduplicate = deduplicate

    def go(self):
        y_train = self.dfAll['relevance'].values
        for obs_field in self.obs_fields:
            if not obs_field in self.dfAll.columns:
                self.logger.info("Skip %s" % (obs_field))
                continue
            if not self.deduplicate:
                obs_corpus = self.dfAll[obs_field].values
            for target_field in self.target_fields:
                if not target_field in self.dfAll.columns:
                    self.logger.info("Skip %s" % (target_field))
                    continue
                if self.deduplicate:
                    combined_corpus = self.dfAll[[obs_field, target_field]].drop_duplicates().values
                    obs_corpus = combined_corpus[:,0]
                    target_corpus = combined_corpus[:,1]
                else:
                    target_corpus = self.dfAll[target_field].values
                estimator = self.generator(obs_corpus, target_corpus, *self.param_list)
                x = estimator.transform()
                if self.deduplicate:
                    x_df = pd.DataFrame(zip(obs_corpus, target_corpus, x), columns=['src1', 'src2', 'est']).set_index(['src1', 'src2'])
                    x = self.dfAll[[obs_field, target_field]].join(x_df, on=[obs_field, target_field], how='left')['est'].values
                    # This feels like a hack, but we have ended up with an ndarray of ndarray on
                    # aggregations and need to fix it
                    if type(x[0]) == np.ndarray:
                        x = np.vstack(x)
                if isinstance(estimator.__name__(), list):
                    for i, feat_name in enumerate(estimator.__name__()):
                        dim = 1
                        fname = "%s_%s_x_%s_%dD" % (feat_name, obs_field, target_field, dim)
                        table_utils._write(os.path.join(self.feat_dir, fname + config.FEAT_FILE_SUFFIX), x[:,i])
                        corr = np_utils._corr(x[:,i], y_train)
                        self.logger.info("%s (%dD): corr = %.6f" % (fname, dim, corr))
                else:
                    dim = np_utils._dim(x)
                    fname = "%s_%s_x_%s_%dD" % (estimator.__name__(), obs_field, target_field, dim)
                    table_utils._write(os.path.join(self.feat_dir, fname + config.FEAT_FILE_SUFFIX), x)
                    if dim == 1:
                        corr = np_utils._corr(x, y_train)
                        self.logger.info("%s (%dD): corr = %.6f" % (fname, dim, corr))


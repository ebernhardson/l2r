import numpy as np
import pandas as pd

import os
import functools

import config
from utils import np_utils, table_utils

# Base class for measuring correlation/similarity/distance
# between the search query and page information
class BaseEstimator(object):
    def __init__(self, obs_corpus, target_corpus, aggregation_mode, id_list=None):
        self.obs_corpus = obs_corpus
        self.N = len(obs_corpus)
        # for standalone features, range so zip works right
        self.target_corpus = range(self.N) if target_corpus is None else target_corpus
        # id_list is used for group based relevance/distance detectors (where?)
        self.id_list = range(self.N) if id_list is None else id_list
        # aggregation for multi-value input fields such as hit_heading and hit_category
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

class BaseMultiEstimatorWrapper(object):
    """
    Base class for wrapping an estimator to support multi-obs or multi-target
    values such as redirect.title and heading
    """

    def __init__(self, generator):
        self.generator = generator

    def __call__(self, *args, **kwargs):
        est = self.generator(*args, **kwargs)
        assert est.aggregation_mode != [""]

        # Evil hax
        name = est.__name__()
        def __name__():
            return ["%s_%s" % (name, x) for x in est.aggregation_mode]
        est.__name__ = __name__

        # It would be nice if we could deduplicate here as well, but
        # it requires pulling the full dataset into memory. A quick
        # test with outgoing links took >10GB before being canceled
        est.transform_one = self.gen_transform_one(est.transform_one)

        return est

class MultiObjEstimatorWrapper(BaseMultiEstimatorWrapper):
    def gen_transform_one(self, transform_one):
        def replacement(obs, target, id):
            assert isinstance(obs, tuple)
            return [transform_one(x, target, id) for x in obs]
        return replacement

class MultiTargetEstimatorWrapper(BaseMultiEstimatorWrapper):
    def gen_transform_one(self, transform_one):
        def replacement(obs, target, id):
            assert isinstance(target, tuple)
            return [transform_one(obs, x, id) for x in target]
        return replacement

def make_transformer(dfAll, field):
    if field in config.ES_DOC_FIELDS:
        transformer = functools.partial(ShelveLookupTransformer, config.ES_PAGE_DOCS_SHELVE, field)
        field = 'hit_page_id'
    elif field[-8:] == '_termvec':
        if field[:-8] in config.ES_TERM_FIELDS:
            es_field = field[:-8]
            fname = config.ES_PAGE_TERM_VEC_SHELVE
            field = 'hit_page_id'
        elif field[:6] == 'query_':
            es_field = field[6:-8]
            fname = config.ES_QUERY_TERM_VEC_SHELVE
            field = 'query'
        elif field[:11] == 'norm_query_':
            es_field = field[11:-8]
            fname = config.ES_QUERY_TERM_VEC_SHELVE
            field = 'norm_query'
        else:
            es_field = None
        if es_field in config.ES_TERM_FIELDS:
            transformer = functools.partial(ShelveLookupTransformer, fname, es_field)
        else:
            transformer = None
    elif not field in dfAll.columns:
        transformer = None
    else:
        transformer = NoopTransformer
    return transformer, field

def make_deduplicator(deduplicate, dfAll, obs_field, target_field):
    if not deduplicate:
        return NoopDeduplicator(dfAll, obs_field, target_field)
    elif target_field is None:
        return SingleFieldDeduplicator(dfAll, obs_field)
    else:
        return DualFieldDeduplicator(dfAll, obs_field, target_field)

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
        self.make_deduplicator = functools.partial(make_deduplicator, deduplicate, dfAll)

    def go(self):
        y_train = self.dfAll["relevance"].values
        for obs_field in self.obs_fields:
            obs_transformer, obs_field_transformed = make_transformer(self.dfAll, obs_field)
            if obs_transformer is None:
                self.logger.info("Skip %s" % (obs_field))
                continue

            deduplicator = self.make_deduplicator(obs_field_transformed, None)
            obs_corpus, _ = deduplicator.deduplicate()
            obs_trans = obs_transformer(obs_corpus)
            estimator = self.generator(obs_trans, None, *self.param_list)
            x = deduplicator.reduplicate(obs_corpus, None, estimator.transform())

            if isinstance(estimator.__name__(), list):
                for i, feat_name in enumerate(estimator.__name__()):
                    self.save_feature(feat_name, obs_field, 1, x[:,i], y_train)
            else:
                dim = np_utils._dim(x)
                self.save_feature(estimator.__name__(), obs_field, dim, x, y_train)

    def save_feature(self, feat_name, obs_field, dim, x, y):
        fname = "%s_%s_%dD" % (feat_name, obs_field, dim)
        table_utils._write(os.path.join(self.feat_dir, fname+config.FEAT_FILE_SUFFIX), x)
        if dim == 1:
            corr = np_utils._corr(x, y)
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
        self.make_deduplicator = functools.partial(make_deduplicator, deduplicate, dfAll)

    def go(self):
        y_train = self.dfAll['relevance'].values
        for obs_field in self.obs_fields:
            obs_transformer, obs_field_transformed = make_transformer(self.dfAll, obs_field)
            if obs_transformer is None:
                self.logger.info("Skip %s" % (obs_field))
                continue

            for target_field in self.target_fields:
                target_transformer, target_field_transformed = make_transformer(self.dfAll, target_field)
                if target_transformer is None:
                    self.logger.info("Skip %s" % (target_field))
                    continue

                deduplicator = self.make_deduplicator(obs_field_transformed, target_field_transformed)
                obs_corpus, target_corpus = deduplicator.deduplicate()
                obs_trans = obs_transformer(obs_corpus)
                target_trans = target_transformer(target_corpus)
                estimator = self.generator(obs_trans, target_trans, *self.param_list)
                x = deduplicator.reduplicate(obs_corpus, target_corpus, estimator.transform())

                if isinstance(estimator.__name__(), list):
                    for i, feat_name in enumerate(estimator.__name__()):
                        self.save_feature(feat_name, obs_field, target_field, 1, x[:,i], y_train)
                else:
                    dim = np_utils._dim(x)
                    self.save_feature(estimator.__name__(), obs_field, target_field, dim, x, y_train)
                # Release memory between iterations. Not sure if necessary yet,
                # but noticed some strange memory usage so trying this out
                del obs_corpus
                del obs_trans
                del target_corpus
                del target_trans
                del x

    def save_feature(self, feat_name, obs_field, target_field, dim, x, y):
        fname = "%s_%s_x_%s_%dD" % (feat_name, obs_field, target_field, dim)
        table_utils._write(os.path.join(self.feat_dir, fname + config.FEAT_FILE_SUFFIX), x)
        if dim == 1:
            corr = np_utils._corr(x, y)
            self.logger.info("%s (%dD): corr = %.6f" % (fname, dim, corr))

class NoopTransformer(object):
    def __init__(self, corpus):
        self.corpus = corpus

    def __len__(self):
        return len(self.corpus)

    def __iter__(self):
        return iter(self.corpus)

    def __getitem__(self, i):
        return self.corpus[i]


# Could be more carefull .. but we will only open at
# most 3 (currently) in read only so whatever...
open_shelves = {}

class ShelveLookupTransformer(object):
    def __init__(self, filename, field, corpus):
        self.filename = filename
        self.field = field
        self.corpus = corpus
        if not filename in open_shelves:
            open_shelves[filename] = table_utils._open_shelve_read(self.filename)
        self.data = open_shelves[filename]

    def __len__(self):
        return len(self.corpus)

    def __iter__(self):
        for key in self.corpus:
            if isinstance(key, unicode):
                val = self.data[key.encode('utf8')]
            else:
                val = self.data[str(key)]
            yield val if self.field is None else val[self.field]

    def __getitem__(self, i):
        key = self.corpus[i]
        if isinstance(key, unicode):
            val = self.data[key.encode('utf8')]
        else:
            val = self.data[str(key)]
        return val if self.field is None else val[self.field]


class NoopDeduplicator(object):
    """
    Fills the deduplicator interface, but does nothing for
    estimators that don't want deduplication
    """
    def __init__(self, dfAll, obs_field, target_field):
        self.dfAll = dfAll
        self.obs_field = obs_field
        self.target_field = target_field

    def deduplicate(self):
        obs_corpus = self.dfAll[self.obs_field].values
        target_corpus = None if self.target_field is None else self.dfAll[self.target_field].value
        return obs_corpus, target_corpus

    def reduplicate(self, obs_corpus, target_corpus, x):
        return x

class SingleFieldDeduplicator(object):
    def __init__(self, dfAll, obs_field):
        self.dfAll = dfAll
        self.obs_field = obs_field

    def deduplicate(self):
        obs_corpus = self.dfAll[self.obs_field].drop_duplicates().values
        return obs_corpus, None

    def reduplicate(self, obs_corpus, target_corpus, x):
        # re-duplicate the values
        x_df = pd.DataFrame(zip(obs_corpus, x), columns=['src', 'est']).set_index(['src'])
        # We need obs_field in a  list to ensure we get back a DataFrame and not a Series
        x_redup = self.dfAll[[self.obs_field]].join(x_df, on=[self.obs_field], how='left')['est'].values
        # This feels like a hack, but we have ended up with an ndarray of ndarray on
        # aggregations and need to fix it
        if type(x[0]) == np.ndarray:
            x_redup = np.vstack(x_redup)
        return x_redup

class DualFieldDeduplicator(object):
    def __init__(self, dfAll, obs_field, target_field):
        self.dfAll = dfAll
        self.obs_field = obs_field
        self.target_field = target_field

    def deduplicate(self):
        combined_corpus = self.dfAll[[self.obs_field, self.target_field]].drop_duplicates().values
        obs_corpus = combined_corpus[:,0]
        target_corpus = combined_corpus[:,1]
        return obs_corpus, target_corpus

    def reduplicate(self, obs_corpus, target_corpus, x):
        x_df = pd.DataFrame(zip(obs_corpus, target_corpus, x), columns=['src1', 'src2', 'est']) \
                .set_index(['src1', 'src2'])
        x_redup = self.dfAll[[self.obs_field, self.target_field]] \
                .join(x_df, on=[self.obs_field, self.target_field], how='left')['est'].values
        # This feels like a hack, but we have ended up with an ndarray of ndarray on
        # aggregations and need to fix it
        if type(x[0]) == np.ndarray:
            x_redup = np.vstack(x_redup)
        return x_redup

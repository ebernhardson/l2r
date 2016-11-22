import numpy as np
import pandas as pd

import config
from utils import logging_utils, table_utils
from feature_base import BaseEstimator, StandaloneFeatureWrapper

class Ident(BaseEstimator):
    """Copy observations from external sources to output"""
    def __init__(self, obs_corpus, target_corpus, transform_func=None, aggregation_mode=""):
        super(Ident, self).__init__(obs_corpus, target_corpus, aggregation_mode)
        self.transform_func = transform_func

    def __name__(self):
        name = "Ident"
        if self.transform_func is not None:
            name += '_' + self.transform_func.__name__.capitalize()
        if isinstance(self.obs_corpus[0], tuple):
            name += '_Len'
        return name

    def transform_one(self, obs, target, id):
        # seems a hack, but termvec sub fields are lists, but multi-value
        # es doc fields are tuples, so we operate on the length of the tuples
        # and allow aggregation to deal with sub fields from termvec
        if isinstance(obs, tuple):
            obs = len(obs)

        if self.transform_func is None:
            return obs
        elif isinstance(obs, list):
            return list(self.transform_func(obs))
        else:
            return self.transform_func(obs)

class SubFieldIdent(Ident):
    """Copy observations from subfield of external source"""
    def __init__(self, obs_corpus, target_corpus, sub_field, transform_func=None, aggregation_mode=""):
        super(SubFieldIdent, self).__init__(obs_corpus, target_corpus, transform_func, aggregation_mode)
        self.sub_field = sub_field

    def __name__(self):
        name = "%s_%s" % (super(SubFieldIdent, self).__name__(), self.sub_field)
        if self.aggregation_mode == [""]:
            return name
        return [name + '_' + x for x in self.aggregation_mode]

    def transform_one(self, obs, target, id):
        return super(SubFieldIdent, self).transform_one(obs[self.sub_field], target, id)

def main():
    logname = "generate_feature_ident.log"
    logger = logging_utils._get_logger(config.LOG_DIR, logname)
    dfAll = table_utils._read(config.ALL_DATA)

    # Copies of data from ES docs. Note that multi-valued fields are first
    # converted into their length
    obs_fields = ["incoming_links", "popularity_score", "text_bytes",
            "category", "template", "heading", "outgoing_link", "external_link",
            "redirect.title", "auxiliary_text"]
    transforms = [None, np.log, np.log10, np.sqrt]
    dedup = True
    for transform in transforms:
        param_list = [transform]
        sf = StandaloneFeatureWrapper(Ident, dfAll, obs_fields, param_list, config.FEAT_DIR, logger, dedup)
        sf.go()

    # Sub-fields from termvec data
    obs_fields = [x + '_termvec' for x in config.ES_TERM_FIELDS]
    obs_fields += ['query_' + x + '_termvec' for x in config.ES_TERM_FIELDS]
    obs_fields += ['norm_query_' + x + '_termvec' for x in config.ES_TERM_FIELDS]
    es_fields = ['score', 'term_freq', 'ttf', 'doc_freq']
    aggregation_mode = ["mean", "std", "max", "min", "median"]
    for es_field in es_fields:
        for transform in transforms:
            param_list = [es_field, transform, aggregation_mode]
            sf = StandaloneFeatureWrapper(SubFieldIdent, dfAll, obs_fields, param_list, config.FEAT_DIR, logger, dedup)
            sf.go()

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np

from itertools import izip

import config
from utils import table_utils, dist_utils
from utils import logging_utils, time_utils
from feature_base import PairwiseFeatureWrapper, BaseEstimator


class ES_TFIDF_Unigram_TopN_CosineSim(BaseEstimator):
    """
    Calculates cosine sim based on the tokenize, stem, score steps
    performed by elasticsearch.
    """
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super(ES_TFIDF_Unigram_TopN_CosineSim, self).__init__(obs_corpus, target_corpus, aggregation_mode)

    def __name__(self):
        return "ES_TFIDF_Unigram_Top%d_CosineSim" % (config.TERM_VEC_MAX_NUM_TERMS)

    def to_term_vec(self, data):
        return dict(zip(data['terms'], data['score']))

    def transform_one(self, obs, target, id):
        obs_vec = self.to_term_vec(obs)
        target_vec = self.to_term_vec(target)
        return dist_utils._es_cosine_sim(obs_vec, target_vec)

def main():
    logname = "generate_feature_tfidf.log"
    logger = logging_utils._get_logger(config.LOG_DIR, logname)

    obs_fields = ['query', 'norm_query']
    target_fields = [x + '_termvec' for x in config.ES_TERM_FIELDS]

    dfAll = table_utils._read(config.ALL_DATA)
    docs = table_utils._open_shelve_read(config.ES_PAGE_TERM_VEC_SHELVE)
    queries = table_utils._open_shelve_read(config.ES_QUERY_TERM_VEC_SHELVE)

    generators = [ES_TFIDF_Unigram_TopN_CosineSim]
    dedup = True
    for generator in generators:
        for target_field in target_fields:
            obs_fields_tv = [x + '_' + target_field for x in obs_fields]
            param_list = []
            # TODO: why iterate obs_fields instead of passing all at once?
            pf = PairwiseFeatureWrapper(generator, dfAll, obs_fields_tv, [target_field], param_list, config.FEAT_DIR, logger, dedup)
            pf.go()

if __name__ == "__main__":
    main()

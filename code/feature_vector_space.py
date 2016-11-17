import pandas as pd
import numpy as np

import config
from utils import table_utils, dist_utils
from utils import logging_utils, time_utils
from feature_base import PairwiseFeatureWrapper

class ES_TFIDF_Word_All_CosineSim:
    """
    Calculates cosine sim based on the tokenize, stem, score steps
    performed by elasticsearch.
    """
    def __init__(self, obs_corpus, target_corpus):
        self.obs_corpus = obs_corpus
        self.target_corpus = target_corpus

    def __name__(self):
        return "ES_TFIDF_Word_All_CosineSim"

    def transform(self):
        sim = list(map(dist_utils._es_cosine_sim, self.obs_corpus, self.target_corpus))
        return np.asarray(sim).squeeze()

def main():
    logname = "generate_feature_tfidf_%s.log" % (time_utils._timestamp())
    logger = logging_utils._get_logger(config.LOG_DIR, logname)

    obs_fields = []
    target_fields = []
    for f in config.ES_TERM_FIELDS:
        obs_fields.append([
            "query_%s_terms" % (f),
            "query_%s_score" % (f),
        ])
        target_fields.append([
            "hit_%s_terms" % (f),
            "hit_%s_score" % (f),
        ])

    # TODO: .drop_duplicates() would make this way faster...but then how would
    # PairwiseFeatureWrapper() join it back to make the proper shape? (pass in
    # a function?)
    dfAll = table_utils._read(config.CLICK_DATA)[['query', 'hit_page_id', 'hit_title']] \
            .join(table_utils._read(config.ES_PAGE_TERM_VEC), on='hit_page_id', how='left') \
            .join(table_utils._read(config.ES_QUERY_TERM_VEC), on='query', how='left') \
            .join(table_utils._read(config.RELEVANCE_DATA).set_index(['query', 'hit_title']), on=['query', 'hit_title'], how='left')

    generators = [ES_TFIDF_Word_All_CosineSim]
    for generator in generators:
        for (obs_field, target_field) in zip(obs_fields, target_fields):
            param_list = []
            pf = PairwiseFeatureWrapper(generator, dfAll, [obs_field], [target_field], param_list, config.FEAT_DIR, logger)
            pf.go()

if __name__ == "__main__":
    main()

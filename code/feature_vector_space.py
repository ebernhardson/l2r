import pandas as pd
import numpy as np

from itertools import izip

import config
from utils import table_utils, dist_utils
from utils import logging_utils, time_utils
from feature_base import PairwiseFeatureWrapper


class ES_TFIDF_Word_All_CosineSim:
    """
    Calculates cosine sim based on the tokenize, stem, score steps
    performed by elasticsearch.
    """
    def __init__(self, obs_corpus, target_corpus, es_field, docs, queries):
        self.obs_corpus = obs_corpus
        self.target_corpus = target_corpus
        self.es_field = es_field
        self.docs = docs
        self.queries = queries

    def __name__(self):
        return "ES_TFIDF_CosineSim_%s" % (self.es_field)

    def to_term_vec(self, raw):
        data = raw[self.es_field]
        return dict(zip(data['terms'], data['score']))

    def prepare(self):
        for query, page_id in izip(self.obs_corpus, self.target_corpus):
            try:
                # Some docs might have been deleted by the time we collected info
                page_data = self.docs[str(page_id)]
            except KeyError:
                page_data = {self.es_field: {'terms':[], 'score':[]}}
            # Query should always exist
            query_data = self.queries[query.encode('utf8')]
            import pprint
            query_term_vec = self.to_term_vec(query_data)
            page_term_vec = self.to_term_vec(page_data)
            yield query_term_vec, page_term_vec
        
    def transform(self):
        # TODO: Can perhaps save memory using a generator, but have to remember
        # why squeeze first
        res = [dist_utils._es_cosine_sim(a, b) for a, b in self.prepare()]
        # TODO: Why squeeze?
        return np.asarray(res).squeeze()

def main():
    logname = "generate_feature_tfidf.log"
    logger = logging_utils._get_logger(config.LOG_DIR, logname)

    obs_fields = ['query', 'norm_query']
    target_fields = ['hit_page_id']

    dfAll = table_utils._read(config.ALL_DATA)
    docs = table_utils._open_shelve_read(config.ES_PAGE_TERM_VEC_SHELVE)
    queries = table_utils._open_shelve_read(config.ES_QUERY_TERM_VEC_SHELVE)

    generators = [ES_TFIDF_Word_All_CosineSim]
    dedup = True
    for generator in generators:
        for es_field in config.ES_TERM_FIELDS:
            param_list = [es_field, docs, queries]
            # TODO: why iterate obs_fields instead of passing all at once?
            pf = PairwiseFeatureWrapper(generator, dfAll, obs_fields, target_fields, param_list, config.FEAT_DIR, logger, dedup)
            pf.go()

if __name__ == "__main__":
    main()

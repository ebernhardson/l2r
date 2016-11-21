import numpy as np
import pandas as pd

from collections import Counter
import re

import config
from utils import ngram_utils, np_utils, nlp_utils
from utils import time_utils, logging_utils, table_utils
from feature_base import BaseEstimator, StandaloneFeatureWrapper, MultiObjEstimatorWrapper

# tune the token pattern to get better correlation, although
# this also varies by language...
token_pattern = " "

class DocLen(BaseEstimator):
    """Number of tokens in document"""
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super(DocLen, self).__init__(obs_corpus, target_corpus, aggregation_mode)

    def __name__(self):
        return "DocLen"

    def transform_one(self, obs, target, id):
        return len(nlp_utils._tokenize(obs, token_pattern))

class DocFreq(BaseEstimator):
    """Frequency of the document in the corpus"""
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super(DocFreq, self).__init__(obs_corpus, target_corpus, aggregation_mode)
        if isinstance(obs_corpus[0], tuple):
            self.counter = Counter((x for y in obs_corpus for x in y))
        else:
            self.counter = Counter(obs_corpus)

    def __name__(self):
        return "DocFreq"

    def transform_one(self, obs, target, id):
        return self.counter[obs]

class DocEntropy(BaseEstimator):
    """Entropy of the document"""
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super(DocEntropy, self).__init__(obs_corpus, target_corpus, aggregation_mode)

    def __name__(self):
        return "DocEntropy"

    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        counter = Counter(obs_tokens)
        count = np.asarray(list(counter.values()))
        proba = count / float(np.sum(count))
        return np_utils._entropy(proba)

class DigitCount(BaseEstimator):
    """Count of digits in the document"""
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super(DigitCount, self).__init__(obs_corpus, target_corpus, aggregation_mode)

    def __name__(self):
        return "DigitCount"

    def transform_one(self, obs, target, id):
        return len(re.findall(r"\d", obs))

class DigitRatio(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super(DigitRatio, self).__init__(obs_corpus, target_corpus, aggregation_mode)

    def __name__(self):
        return "DigitRatio"

    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        digits = re.findall(r"\d", obs)
        return np_utils._try_divide(len(digits), len(obs_tokens))

class UniqueCount_Ngram(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode=""):
        super(UniqueCount_Ngram, self).__init__(obs_corpus, target_corpus, aggregation_mode)
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]

    def __name__(self):
        return "UniqueCount_%s" % (self.ngram_str)

    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        obs_ngrams = ngram_utils._ngrams(obs_tokens, self.ngram)
        return len(set(obs_ngrams))

class UniqueRatio_Ngram(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode=""):
        super(UniqueRatio_Ngram, self).__init__(obs_corpus, target_corpus, aggregation_mode)
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]

    def __name__(self):
        return "UniqueRatio_%s" % (self.ngram_str)

    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        obs_ngrams = ngram_utils._ngrams(obs_tokens, self.ngram)
        return np_utils._try_divide(len(set(obs_ngrams)), len(obs_ngrams))

def main():
    logname = "generate_feature_basic.log"
    logger = logging_utils._get_logger(config.LOG_DIR, logname)
    dfAll = table_utils._read(config.ALL_DATA)

    # basic
    generators = [DocLen, DocFreq, DocEntropy, DigitCount, DigitRatio]
    obs_fields = ["query", "norm_query", "hit_title", 'opening_text']
    for generator in generators:
        param_list = []
        dedup = False if generator == DocFreq else True
        sf = StandaloneFeatureWrapper(generator, dfAll, obs_fields, param_list, config.FEAT_DIR, logger, dedup)
        sf.go()

    # basic against multi-value fields
    obs_fields = ['category', 'template', 'heading',
            'outgoing_link', 'external_link', 'redirect.title',
            'auxiliary_text']
    aggregations = ['mean', 'std', 'max', 'min', 'median']
    param_list = [aggregations]
    for generator in generators:
        multi_gen = MultiObjEstimatorWrapper(generator)
        dedup = False if generator == DocFreq else True
        sf = StandaloneFeatureWrapper(multi_gen, dfAll, obs_fields, param_list, config.FEAT_DIR, logger, dedup)
        sf.go()

    # unique count
    generators = [UniqueCount_Ngram, UniqueRatio_Ngram]
    obs_fields = ["query", "norm_query", "hit_title", 'opening_text']
    ngrams = [1,2,3]
    for generator in generators:
        for ngram in ngrams:
            param_list = [ngram]
            dedup = True
            sf = StandaloneFeatureWrapper(generator, dfAll, obs_fields, param_list, config.FEAT_DIR, logger, dedup)
            sf.go()

    # unique count against multi-value fields
    generators = [UniqueCount_Ngram, UniqueRatio_Ngram]
    obs_fields = ['category', 'template', 'heading',
            'outgoing_link', 'external_link', 'redirect.title',
            'auxiliary_text']
    aggregations = ['mean', 'std', 'max', 'min', 'median']
    ngrams = [1,2,3]
    for generator in generators:
        for ngram in ngrams:
            multi_gen = MultiObjEstimatorWrapper(generator)
            param_list = [ngram, aggregations]
            dedup = True
            sf = StandaloneFeatureWrapper(multi_gen, dfAll, obs_fields, param_list, config.FEAT_DIR, logger, dedup)
            sf.go()

if __name__ == "__main__":
    main()

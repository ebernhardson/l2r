import numpy as np
import pandas as pd
from collections import Counter

import config
from utils import ngram_utils, np_utils
from utils import time_utils, logging_utils, msgpack_utils
from feature_base import BaseEstimator, StandaloneFeatureWrapper

class DocLen(BaseEstimator):
    """Number of tokens in document"""
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)

    def __name__(self):
        return "DocLen"

    def transform_one(self, obs, target, id):
        return len(obs)

class DocFreq(BaseEstimator):
    """Frequency of the document in the corpus"""
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        self.counter = Counter(obs_corpus)

    def __name__(self):
        return "DocFreq"

    def transform_one(self, obs, target, id):
        return self.counter[obs]


class DocEntropy(BaseEstimator):
    """Entropy of the document"""
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)

    def __name__(self):
        return "DocEntropy"

    def transform_one(self, obs, target, id):
        counter = Counter(obs)
        count = np.asarray(list(counter.values()))
        proba = count / np.sum(count)
        return np_utils._entropy(proba)

class DigitCount(BaseEstimator):
    """Count of digits in the document"""
    def __init__(self, obs_corpus, target_corpus, aggregation_mode="")
        super().__init__(obs_corpus, target_corpus, aggregation_mode)

    def __name__(self):
        return "DigitCount"

    def transform_one(self, obs, target, id):
        return len(re.findall(r"\d", obs))

class DigitRatio(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, aggregation_mode="")
        super().__init__(obs_corpus, target_corpus, aggregation_mode)

    def __name__(self):
        return "DigitRatio"

    def transform_one(self, obs, target, id):
        return np_utils._try_divide(len(re.findall(r"\d", obs)), len(obs))

class UniqueCount_Ngram(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode="")
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]

    def __name__(self):
        return "UniqueCount_%s" % (self.ngram_str)

    def transform_one(self, obs, target, id):
        obs_ngrams = ngrams_utils._ngrams(obs, self.ngram)
        return len(set(obs_ngrams))

class UniqueRatio_Ngram(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode="")
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]

    def __name__(self):
        return "UniqueRatio_%s" % (self.ngram_str)

    def transform_one(self, obs, target, id):
        obs_ngrams = ngram_utils._ngrams(obs, self.ngram)
        return np_utils._try_divide(len(set(obs_ngrams)), len(obs_ngrams))

def main():
    logname = "generate_feature_basic_%s.log" % (time_utils._timestamp())
    logger = logging_utils._get_logger(config.LOG_DIR, logname)
    dfAll = msgpack_utils._load(config.ALL_DATA_STEMMED)

    # basic
    generators = [DocLen, DocFreq, DocEntropy, DigitCount, DigitRatio]
    obs_fields = ["search_term", "page_title", "page_text", "page_category",
            "page_template", "page_heading", "page_outgoing_link",
            "page_external_link", "page_redirect_title", "page_auxiliary_text",
            "page_opening_text"]
    for generator in generators:
        param_list = []
        sf = StandaloneFeatureWrapper(generator, dfAll, obs_fields, param_list, config.FEAT_DIR, logger)
        sf.go()

    # unique count
    generators = [UniqueCount_Ngram, UniqueRatio_Ngram]
    ngrams = [1,2,3]
    for generator in generators:
        for ngram in ngrams:
            param_list = [n_gram]
            sf = StandaloneFeatureWrapper(generator, dfAll, obs_fields, param_list, config.FEAT_DIR, logger)
            sf.go()


import numpy as np
import pandas as pd

from collections import Counter
import re

import config
from utils import ngram_utils, np_utils, nlp_utils
from utils import time_utils, logging_utils, table_utils
from feature_base import BaseEstimator, StandaloneFeatureWrapper

# tune the token pattern to get better correlation, although
# this also varies by language...
token_pattern = " "

class DocLen(BaseEstimator):
    """Number of tokens in document"""
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super(DocLen, self).__init__(obs_corpus, target_corpus, aggregation_mode)

    def __name__(self):
        if len(self.aggregation_mode) > 1:
            return ["DocLen_%s" % (x) for x in self.aggregation_mode]
        else:
            return "DocLen"

    def transform_one(self, obs, target, id):
        is_agg = len(self.aggregation_mode) > 1
        if is_agg:
            assert isinstance(obs, tuple)
            obs_list = obs
        else:
            obs_list = [obs]
        res = [len(nlp_utils._tokenize(x, token_pattern)) for x in obs_list]
        return res if is_agg else res[0]

class DocFreq(BaseEstimator):
    """Frequency of the document in the corpus"""
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super(DocFreq, self).__init__(obs_corpus, target_corpus, aggregation_mode)
        if isinstance(obs_corpus[0], tuple):
            self.counter = Counter((x for y in obs_corpus for x in y))
        else:
            self.counter = Counter(obs_corpus)

    def __name__(self):
        if len(self.aggregation_mode) > 1:
            return ["DocFreq_%s" % (x) for x in self.aggregation_mode]
        else:
            return "DocFreq"

    def transform_one(self, obs, target, id):
        is_agg = len(self.aggregation_mode) > 1
        if is_agg:
            assert isinstance(obs, tuple)
            obs_list = obs
        else:
            obs_list = [obs]
        res = [self.counter[x] for x in obs_list]
        return res if is_agg else res[0]

class DocEntropy(BaseEstimator):
    """Entropy of the document"""
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super(DocEntropy, self).__init__(obs_corpus, target_corpus, aggregation_mode)

    def __name__(self):
        if len(self.aggregation_mode) > 1:
            return ["DocEntropy_%s" % (x) for x in self.aggregation_mode]
        else:
            return "DocEntropy"

    def transform_one(self, obs, target, id):
        is_agg = len(self.aggregation_mode) > 1
        if is_agg:
            assert isinstance(obs, tuple)
            obs_list = obs
        else:
            obs_list = [obs]
        res = []
        for one_obs in obs_list:
            obs_tokens = nlp_utils._tokenize(one_obs, token_pattern)
            counter = Counter(obs_tokens)
            count = np.asarray(list(counter.values()))
            proba = count / float(np.sum(count))
            res.append(np_utils._entropy(proba))
        return res if is_agg else res[0]

class DigitCount(BaseEstimator):
    """Count of digits in the document"""
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super(DigitCount, self).__init__(obs_corpus, target_corpus, aggregation_mode)

    def __name__(self):
        if len(self.aggregation_mode) > 1:
            return ["DigitCount_%s" % (x) for x in self.aggregation_mode]
        else:
            return "DigitCount"

    def transform_one(self, obs, target, id):
        is_agg = len(self.aggregation_mode) > 1
        if is_agg:
            assert isinstance(obs, tuple)
            obs_list = obs
        else:
            obs_list = [obs]
        res = [len(re.findall(r"\d", x)) for x in obs_list]
        return res if is_agg else res[0]

class DigitRatio(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super(DigitRatio, self).__init__(obs_corpus, target_corpus, aggregation_mode)

    def __name__(self):
        if len(self.aggregation_mode) > 1:
            return ["DigitRatio_%s" % (x) for x in self.aggregation_mode]
        else:
            return "DigitRatio"

    def transform_one(self, obs, target, id):
        is_agg = len(self.aggregation_mode) > 1
        if is_agg:
            assert isinstance(obs, tuple)
            obs_list = obs
        else:
            obs_list = [obs]
        res = []
        for one_obs in obs_list:
            obs_tokens = nlp_utils._tokenize(one_obs, token_pattern)
            digits = re.findall(r"\d", one_obs)
            res.append(np_utils._try_divide(len(digits), len(obs_tokens)))
        return res if is_agg else res[0]

class UniqueCount_Ngram(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode=""):
        super(UniqueCount_Ngram, self).__init__(obs_corpus, target_corpus, aggregation_mode)
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]

    def __name__(self):
        name = "UniqueCount_%s" % (self.ngram_str)
        if len(self.aggregation_mode) > 1:
            return ["%s_%s" % (name, x) for x in self.aggregation_mode]
        else:
            return name

    def transform_one(self, obs, target, id):
        is_agg = len(self.aggregation_mode) > 1
        if is_agg:
            assert isinstance(obs, tuple)
            obs_list = obs
        else:
            obs_list = [obs]
        res = []
        for one_obs in obs_list:
            obs_tokens = nlp_utils._tokenize(one_obs, token_pattern)
            obs_ngrams = ngram_utils._ngrams(obs_tokens, self.ngram)
            res.append(len(set(obs_ngrams)))
        return res if is_agg else res[0]

class UniqueRatio_Ngram(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode=""):
        super(UniqueRatio_Ngram, self).__init__(obs_corpus, target_corpus, aggregation_mode)
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]

    def __name__(self):
        name = "UniqueRatio_%s" % (self.ngram_str)
        if len(self.aggregation_mode) > 1:
            return ["%s_%s" % (name, x) for x in self.aggregation_mode]
        else:
            return name

    def transform_one(self, obs, target, id):
        is_agg = len(self.aggregation_mode) > 1
        if is_agg:
            assert isinstance(obs, tuple)
            obs_list = obs
        else:
            obs_list = [obs]
        res = []
        for one_obs in obs_list:
            obs_tokens = nlp_utils._tokenize(one_obs, token_pattern)
            obs_ngrams = ngram_utils._ngrams(obs_tokens, self.ngram)
            res.append(np_utils._try_divide(len(set(obs_ngrams)), len(obs_ngrams)))
        return res if is_agg else res[0]

def main():
    logname = "generate_feature_basic.log"
    logger = logging_utils._get_logger(config.LOG_DIR, logname)
    dfAll = table_utils._read(config.ALL_DATA)

    # TODO: This is incredibly wasteful, needing tons of memory
    # to pull in many duplicates of data because there are on average
    # 75+ search sessions with the same hit_title/opening_text/category/etc.

    # basic
    generators = [DocLen, DocFreq, DocEntropy, DigitCount, DigitRatio]
    obs_fields = ["query", "norm_query", "hit_title", 'hit_opening_text']
    for generator in generators:
        param_list = []
        dedup = False if generator == DocFreq else True
        sf = StandaloneFeatureWrapper(generator, dfAll, obs_fields, param_list, config.FEAT_DIR, logger, dedup)
        sf.go()

    # basic against multi-value fields
    obs_fields = ['hit_category', 'hit_template', 'hit_heading',
            'hit_outgoing_link', 'hit_external_link', 'hit_redirect.title',
            'hit_auxiliary_text']
    aggregations = ['mean', 'std', 'max', 'min', 'median']
    for generator in generators:
        param_list = [aggregations]
        dedup = False if generator == DocFreq else True
        sf = StandaloneFeatureWrapper(generator, dfAll, obs_fields, param_list, config.FEAT_DIR, logger, dedup)
        sf.go()

    # unique count
    generators = [UniqueCount_Ngram, UniqueRatio_Ngram]
    obs_fields = ["query", "norm_query", "hit_title", 'hit_opening_text']
    ngrams = [1,2,3]
    for generator in generators:
        for ngram in ngrams:
            param_list = [ngram]
            dedup = True
            sf = StandaloneFeatureWrapper(generator, dfAll, obs_fields, param_list, config.FEAT_DIR, logger, dedup)
            sf.go()

    # unique count against multi-value fields
    generators = [UniqueCount_Ngram, UniqueRatio_Ngram]
    obs_fields = ['hit_category', 'hit_template', 'hit_heading',
            'hit_outgoing_link', 'hit_external_link', 'hit_redirect.title',
            'hit_auxiliary_text']
    aggregations = ['mean', 'std', 'max', 'min', 'median']
    ngrams = [1,2,3]
    for generator in generators:
        for ngram in ngrams:
            param_list = [ngram, aggregations]
            dedup = True
            sf = StandaloneFeatureWrapper(generator, dfAll, obs_fields, param_list, config.FEAT_DIR, logger, dedup)
            sf.go()

if __name__ == "__main__":
    main()

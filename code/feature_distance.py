# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: distance features

"""

import re
import sys
import string

import numpy as np
import pandas as pd

import config
from utils import dist_utils, ngram_utils, nlp_utils
from utils import logging_utils, time_utils, table_utils
from feature_base import BaseEstimator, PairwiseFeatureWrapper


# tune the token pattern to get a better correlation with y_train
# token_pattern = r"(?u)\b\w\w+\b"
# token_pattern = r"\w{1,}"
# token_pattern = r"\w+"
# token_pattern = r"[\w']+"
token_pattern = " " # just split the text into tokens


# ------------------- Jaccard & Dice --------------------------------------
class JaccardCoef_Ngram(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode=""):
        super(JaccardCoef_Ngram, self).__init__(obs_corpus, target_corpus, aggregation_mode)
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]
        self.is_multi_target = self.aggregation_mode != [""]

    def __name__(self):
        name = "JaccardCoef_%s"%self.ngram_str
        if self.is_multi_target:
            return ["%s_%s" %(name, agg_mode) for agg_mode in self.aggregation_mode]
        else:
            return name

    def transform_one(self, obs, target, id):
        assert isinstance(obs, unicode)
        if self.is_multi_target:
            assert isinstance(target, tuple)
            target_list = target
        else:
            assert isinstance(target, unicode)
            target_list = [target]
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        obs_ngrams = ngram_utils._ngrams(obs_tokens, self.ngram)
        res = []
        for one_target in target_list:
            target_tokens = nlp_utils._tokenize(one_target, token_pattern)
            target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
            res.append(dist_utils._jaccard_coef(obs_ngrams, target_ngrams))
        return res if self.is_multi_target else res[0]

class DiceDistance_Ngram(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode=""):
        super(DiceDistance_Ngram, self).__init__(obs_corpus, target_corpus, aggregation_mode)
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]
        self.is_multi_target = self.aggregation_mode != [""]

    def __name__(self):
        name = "DiceDistance_%s"%self.ngram_str
        if self.is_multi_target:
            return ["%s_%s" %(name, agg_mode) for agg_mode in self.aggregation_mode]
        else:
            return name

    def transform_one(self, obs, target, id):
        assert isinstance(obs, unicode)
        if self.is_multi_target:
            assert isinstance(target, tuple)
            target_list = obs
        else:
            assert isinstance(target, unicode)
            target_list = [target]
        res = []
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        obs_ngrams = ngram_utils._ngrams(obs_tokens, self.ngram)
        for one_target in target_list:
            target_tokens = nlp_utils._tokenize(one_target, token_pattern)
            target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
            res.append(dist_utils._dice_dist(obs_ngrams, target_ngrams))
        return res if self.is_multi_target else res[0]

# ------------------ Edit Distance --------------------------------
class EditDistance(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super(EditDistance, self).__init__(obs_corpus, target_corpus, aggregation_mode)
        self.is_multi_target = self.aggregation_mode != [""]
    
    def __name__(self):
        name = "EditDistance"
        if self.is_multi_target:
            return ["%s_%s" %(name, agg_mode) for agg_mode in self.aggregation_mode]
        else:
            return name

    def transform_one(self, obs, target, id):
        if self.is_multi_target:
            assert isinstance(target, tuple)
            target_list = obs
        else:
            assert isinstance(target, unicode)
            target_list = [target]
        res = [dist_utils._edit_dist(obs, x) for x in target_list]
        return res if self.is_multi_target else res[0]


# ---------------------------- Main --------------------------------------
def run_ngram_jaccard():
    logname = "generate_feature_ngram_jaccard.log"
    logger = logging_utils._get_logger(config.LOG_DIR, logname)
    dfAll = table_utils._read(config.ALL_DATA)

    generators = [JaccardCoef_Ngram, DiceDistance_Ngram]
    # single valued fields
    obs_fields_list = [["query", "norm_query"]]
    target_fields_list = [["hit_title", "hit_opening_text" ]]
    ngrams = [1,2,3,12,123][:3]
    dedup = True
    for obs_fields, target_fields in zip(obs_fields_list, target_fields_list):
        for generator in generators:
            for ngram in ngrams:
                param_list = [ngram]
                pf = PairwiseFeatureWrapper(generator, dfAll, obs_fields, target_fields, param_list, config.FEAT_DIR, logger, dedup)
                pf.go()

    # multi-valued fields
    target_fields_list = [["hit_category", "hit_template", "hit_heading",
            "hit_outgoing_link", "hit_external_link", "hit_redirect.title",
            "hit_auxiliary_text"]]
    aggregation_mode = ["mean", "std", "max", "min", "median"]
    for obs_fields, target_fields in zip(obs_fields_list, target_fields_list):
        for generator in generators:
            for ngram in ngrams:
                param_list = [ngram, aggregation_mode]
                pf = PairwiseFeatureWrapper(generator, dfAll, obs_fields, target_fields, param_list, config.FEAT_DIR, logger, dedup)
                pf.go()

def run_edit_distance():
    logname = "generate_feature_edit_distance.log"
    logger = logging_utils._get_logger(config.LOG_DIR, logname)
    dfAll = table_utils._read(config.ALL_DATA)

    # single value targets
    obs_fields_list = [["query", "norm_query"]]
    target_fields_list = [["hit_title", "hit_opening_text"]]
    ngrams = [1,2,3,12,123][:3]
    dedup = True
    for obs_fields, target_fields in zip(obs_fields_list, target_fields_list):
        param_list = []
        pf = PairwiseFeatureWrapper(EditDistance, dfAll, obs_fields, target_fields, param_list, config.FEAT_DIR, logger, dedup)
        pf.go()

    # multi-value targets
    target_fields_list = [["hit_category", "hit_template", "hit_heading",
            "hit_outgoing_link", "hit_external_link", "hit_redirect.title",
            "hit_auxiliary_text"]]
    aggregation_mode = ["mean", "std", "max", "min", "median"]
    for obs_fields, target_fields in zip(obs_fields_list, target_fields_list):
        param_list = [aggregation_mode]
        pf = PairwiseFeatureWrapper(EditDistance, dfAll, obs_fields, target_fields, param_list, config.FEAT_DIR, logger, dedup)
        pf.go()

if __name__ == "__main__":
    run_ngram_jaccard()
    run_edit_distance()

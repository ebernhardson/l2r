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
from feature_base import BaseEstimator, PairwiseFeatureWrapper, MultiTargetEstimatorWrapper


# tune the token pattern to get a better correlation with y_train
# token_pattern = r"(?u)\b\w\w+\b"
# token_pattern = r"\w{1,}"
# token_pattern = r"\w+"
# token_pattern = r"[\w']+"
token_pattern = " " # just split the text into tokens


# ------------------- Jaccard & Dice --------------------------------------
class BaseDistance(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode=""):
        super(BaseDistance, self).__init__(obs_corpus, target_corpus, aggregation_mode)
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]
        self.last_obs = None

    def transform_one(self, obs, target, id):
        assert isinstance(obs, unicode)
        assert isinstance(target, unicode)

        if obs != self.last_obs:
            self.last_obs = obs
            obs_tokens = nlp_utils._tokenize(obs, token_pattern)
            self.last_obs_ngrams = ngram_utils._ngrams(obs_tokens, self.ngram)
        target_tokens = nlp_utils._tokenize(target, token_pattern)
        target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
        return self.distance(self.last_obs_ngrams, target_ngrams)

class JaccardCoef_Ngram(BaseDistance):
    def __name__(self):
        return "JaccardCoef_%s" % (self.ngram_str)

    def distance(self, obs_ngrams, target_ngrams):
        return dist_utils._jaccard_coef(obs_ngrams, target_ngrams)


class DiceDistance_Ngram(BaseDistance):
    def __name__(self):
        return "DiceDistance_%s" % (self.ngram_str)

    def distance(self, obs_ngrams, target_ngrams):
        return dist_utils._dice_dist(obs_ngrams, target_ngrams)

# ------------------ Edit Distance --------------------------------
class EditDistance(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super(EditDistance, self).__init__(obs_corpus, target_corpus, aggregation_mode)
    
    def __name__(self):
        return "EditDistance"

    def transform_one(self, obs, target, id):
        return dist_utils._edit_dist(obs, target)

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
            multi_gen = MultiTargetEstimatorWrapper(generator)
            for ngram in ngrams:
                param_list = [ngram, aggregation_mode]
                pf = PairwiseFeatureWrapper(multi_gen, dfAll, obs_fields, target_fields, param_list, config.FEAT_DIR, logger, dedup)
                pf.go()

def run_edit_distance():
    logname = "generate_feature_edit_distance.log"
    logger = logging_utils._get_logger(config.LOG_DIR, logname)
    dfAll = table_utils._read(config.ALL_DATA)

    # single value targets
    obs_fields_list = [["query", "norm_query"]]
    target_fields_list = [["hit_title", "hit_opening_text"]]
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
        multi_gen = MultiTargetEstimatorWrapper(EditDistance)
        pf = PairwiseFeatureWrapper(multi_gen, dfAll, obs_fields, target_fields, param_list, config.FEAT_DIR, logger, dedup)
        pf.go()

if __name__ == "__main__":
    run_ngram_jaccard()
    run_edit_distance()

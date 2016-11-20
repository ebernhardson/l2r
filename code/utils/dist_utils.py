import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

import sys
import math
from difflib import SequenceMatcher
import Levenshtein

from utils import np_utils
sys.path.append("..")
import config

def _es_cosine_sim(a, b):
    # Calculates cosine sim based on tokenize, stem, score
    # generated externally in elasticsearch. This has the
    # downside of only using top n tokens for large fields

    if type(a[0]) == float or type(b[0]) == float:
        # Should have been list-ish, these are nan's
        return 0.0

    # args are in the form [[word, word], [score, score]]
    # building a sparse matrix and shoving through sklearn
    # cosine sim might(?) be faster, but this was more obvious
    a = dict(zip(*list(a)))
    b = dict(zip(*list(b)))

    intersect = set(a.keys()).intersection(b.keys())
    numerator = sum(a[x] * b[x] for x in intersect)

    a_len = sum(x**2 for x in a.values())
    b_len = sum(x**2 for x in b.values())
    denominator = math.sqrt(a_len) * math.sqrt(b_len)

    if denominator:
        return float(numerator) / denominator
    else:
        return 0.0

def _edit_dist(str1, str2):
    try:
        # very fast
        # http://stackoverflow.com/questions/14260126/how-python-levenshtein-ratio-is-computed
        # d = Levenshtein.ratio(str1, str2)
        d = Levenshtein.distance(str1, str2)/float(max(len(str1),len(str2)))
    except:
        # https://docs.python.org/2/library/difflib.html
        d = 1. - SequenceMatcher(lambda x: x==" ", str1, str2).ratio()
    return d

def _jaccard_coef(A, B):
    if not isinstance(A, set):
        A = set(A)
    if not isinstance(B, set):
        B = set(B)
    return np_utils._try_divide(float(len(A.intersection(B))), len(A.union(B)))

def _dice_dist(A, B):
    if not isinstance(A, set):
        A = set(A)
    if not isinstance(B, set):
        B = set(B)
    return np_utils._try_divide(2.*float(len(A.intersection(B))), (len(A) + len(B)))

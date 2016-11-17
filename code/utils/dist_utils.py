import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

import sys
import math

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

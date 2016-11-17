import pandas as pd
import numpy as np

import math

import config
from utils import np_utils, table_utils

def main():
    dfAll = table_utils._read(config.ALL_DATA)

    # Single value fields
    funcs = {
        'id': id
        'log': math.log,
        'log1p': math.log1p,
        'log10': math.log10,
        'sqrt': math.sqrt,
    }

    y_train = dfAll["relevance"].values
    for field in config.ES_DOC_FIELDS:
        for fname, func in funcs.iteritems():
            x = dfAll[field].apply(func).values
            corr = np_utils._corr(x, y_train)
            print "%s %s: %.6f" % (field, fname, corr)

    # multi-value fields
    funcs = {
        'sum': np.nansum,
        'mean': np.nanmean,
        'len': len,
        'loglen':  lambda x: math.log(len(x)),
        'std': np.nanstd,
        'max': np.nanmax,
        'min': np.nanmin,
        'median': np.nanmedian
    }
    for field in config.ES_TERM_FIELDS:
        for fname, func in funcs.iteritems():
            for x in ['query', 'hit']:
                for y in ['score', 'term_freq', 'ttf', 'doc_freq']:
                    f = "%s_%s_%s" % (x, field, y)
                    x = dfAll[f].apply(func).values
                    corr = np_utils._corr(x, y_train)
                    print "%s %s: %.6f" % (field, fname, corr)


    # TODO: string fields

if __name__ == "__main__":


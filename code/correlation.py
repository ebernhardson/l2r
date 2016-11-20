from __future__ import print_function
import pandas as pd
import numpy as np
from scipy import stats

import math

import config
from utils import np_utils, table_utils

def main():
            #.join(table_utils._read(config.ES_PAGE_TERM_VEC), on='hit_page_id', how='left') \
            #.join(table_utils._read(config.ES_QUERY_TERM_VEC), on='query', how='left') \
    # This ends up also including the weight of many sessions against same (query, hit_title)
    # pairs in the correlation. Is that desirable?
    dfAll = table_utils._read(config.CLICK_DATA) \
            .join(table_utils._read(config.ES_PAGE_DOCS), on='hit_page_id', how='left') \
            .join(table_utils._read(config.RELEVANCE_DATA).set_index(['norm_query', 'hit_title']), on=['norm_query', 'hit_title'], how='left')

    #dfAll = dfAll[~dfAll['incoming_links'].isnull()]
    # Single value fields
    single_funcs = {
        'id(%s)': lambda x: x,
        'log(%s)': np.log,
        'log2(%s)': np.log2,
        'log1p(%s)': np.log1p,
        'log10(%s)': np.log10,
        'sqrt(%s)': np.sqrt,
        'square(%s)': np.square,
        'exp(%s)': np.exp,
        'expm1(%s)': np.expm1,
        'log(%s+2)': lambda x: np.log(x + 2),
        'log(%s+1e-7)': lambda x: np.log(x + 1e-7),
        'log(1-%s)': lambda x: np.log(1 - x),
    }

    corr_name = 'pearsonr'
    def corr_func(x, y):
        return stats.pearsonr(x, y)[0]

    pairs = []
    dfAll = dfAll[~dfAll['relevance'].isnull()]

    # Correlations are much stronger when removing the "wall" at .24 <= x <= .25
    # dfAll = dfAll[(dfAll['relevance'] > 0.25) | (dfAll['relevance'] < 0.24)]
    for field in ['hit_incoming_links', 'hit_popularity_score', 'hit_text_bytes']:
        for x_fname, x_func in single_funcs.iteritems():
            dfClean = dfAll[~dfAll[field].isnull()]
            xt = dfClean[field].apply(x_func).values
            for y_fname, y_func in single_funcs.iteritems():
                yt = dfClean["relevance"].apply(y_func).values
                corr = corr_func(xt.flatten(), yt)
                if not np.isnan(corr):
                    pairs.append((abs(corr), "%s(%s, %s): %.6f" % (corr_name, x_fname % field, y_fname % 'relevance', corr)))
                print(".", end="")

    print("\n")
    for _, s in sorted(pairs, key=lambda tup: tup[0]):
        print(s)

    # multi-value fields
    #funcs = {
    #    'sum': np.nansum,
    #    'mean': np.nanmean,
    #    'len': len,
    #    'loglen':  lambda x: math.log(len(x)),
    #    'std': np.nanstd,
    #    'max': np.nanmax,
    #    'min': np.nanmin,
    #    'median': np.nanmedian
    #}
    #for field in config.ES_TERM_FIELDS:
    #    for fname, func in funcs.iteritems():
    #        for x in ['query', 'hit']:
    #            for y in ['score', 'term_freq', 'ttf', 'doc_freq']:
    #                f = "%s_%s_%s" % (x, field, y)
    #                x = dfAll[f].apply(func).values
    #                corr = np_utils._corr(x, y_train)
    #                print("%s %s: %.6f" % (field, fname, corr))


    # TODO: string fields

if __name__ == "__main__":
    main()

import pandas as pd

import config
from utils import table_utils

def dfRel():
    return table_utils._read(config.RELEVANCE_DATA)

def shelve_keys(fname):
    data = table_utils._open_shelve_read(fname)
    keys = data.keys()
    data.close()
    return keys

def main():
    # uses inner to ensure we have complete rows. Otherwise
    # we would have NaN mixed in with the relevance scores,
    # and we can't train on those
    dfClicks = table_utils._read(config.CLICK_DATA)
    dfAll = dfClicks \
            .join(dfRel().set_index(['norm_query', 'hit_title']),
                  on=['norm_query', 'hit_title'], how='inner')

    # Filter out pages that couldn't be loaded as docs/termvecs
    es_docs_keys = set(map(int, shelve_keys(config.ES_PAGE_DOCS_SHELVE)))
    dfAll = dfAll[dfAll['hit_page_id'].isin(es_docs_keys)]

    es_termvec_keys = set(map(int, shelve_keys(config.ES_PAGE_TERM_VEC_SHELVE)))
    dfAll = dfAll[dfAll['hit_page_id'].isin(es_termvec_keys)]


    table_utils._write(config.ALL_DATA, dfAll)

    print 'Source clicks len: %d' % (len(dfClicks))
    print 'Final data len: %d' % (len(dfAll))
    print 'Ratio: %.3f' % (float(len(dfAll))/len(dfClicks))


if __name__ == "__main__":
    main()

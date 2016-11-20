import pandas as pd

import config
from utils import table_utils

# Helper methods make it a bit cleaner to just pull in the df
# once for a join, and let gc throw it back out, hopefully
def dfRel():
    return table_utils._read(config.RELEVANCE_DATA)

def dfEsPageDocs():
    return table_utils._read(config.ES_PAGE_DOCS)

def dfEsPageTermVec():
    return table_utils._read(config.ES_PAGE_TERM_VEC)

def dfEsQueryTermVec():
    return table_utils._read(config.ES_QUERY_TERM_VEC)

def main():
    # Consistently uses inner to ensure we have complete rows. Otherwise
    # we would have oddities like mixed unicode and NaN
    dfClicks = table_utils._read(config.CLICK_DATA)
    dfAll = dfClicks \
            .join(dfRel().set_index(['norm_query', 'hit_title']),
                          on=['norm_query', 'hit_title'], how='inner') \
            .join(dfEsPageDocs(), on=['hit_page_id'], how='inner')
            # These are rediculous memory hogs, although so is the above...
            # Need to find a better way to generate features without
            # all the duplication and resulting memory explosion
            #.join(dfEsPageTermVec(), on='hit_page_id', how='left') \
            #.join(dfEsQueryTermVec(), on='query', how='left')

    table_utils._write(config.ALL_DATA, dfAll)

    print 'Source clicks len: %d' % (len(dfClicks))
    print 'Final data len: %d' % (len(dfAll))
    print 'Ratio: %.3f' % (float(len(dfAll))/len(dfClicks))


if __name__ == "__main__":
    main()

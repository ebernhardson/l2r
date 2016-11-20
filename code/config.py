from utils import os_utils

# Top level configuration
TASK = "all"

ES_URL = 'http://elastic1020.eqiad.wmnet:9200/enwiki_content'

# query options for the source data
# Number of times the query must have been issued in 1 week
MIN_NUM_SEARCHES = 35
# Project searched against
WIKI_PROJECT = 'en.wikipedia'
# The number of unique queries to source
MAX_QUERIES = 100

# Identifier of the query being worked on
QUERY_IDENT = "%s_%dS_%dQ" % (WIKI_PROJECT, MIN_NUM_SEARCHES, MAX_QUERIES)

# A few basic directories
ROOT_DIR = "../"
DATA_DIR = "%s/data/%s" % (ROOT_DIR, QUERY_IDENT)
FEAT_DIR = "%s/features/%s" % (ROOT_DIR, QUERY_IDENT)
LOG_DIR = "%s/log/%s" % (ROOT_DIR, QUERY_IDENT)
TMP_DIR = "%s/tmp" % (ROOT_DIR)

# elasticsearch doc fields to augment with
ES_DOC_FIELDS_DEFAULTS = {
    'incoming_links': 0,
    'popularity_score': 0,
    'text_bytes': 0,
    'wp10': 0,
    # Giving all the defaults a single value allows features
    # to always assume a value exists. These must be tuples
    # for drop_duplicates() to work
    'category': (u"",),
    'template': (u"",),
    'heading': (u"",),
    'outgoing_link': (u"",),
    'external_link': (u"",),
    'redirect.title': (u"",),
    'auxiliary_text': (u"",),
    'opening_text': u"",
}
ES_DOC_FIELDS = ES_DOC_FIELDS_DEFAULTS.keys()

# fields to collect term vectors for
ES_TERM_FIELDS = [
    'title', 'title.plain',
    'heading', 'heading.plain',
    'opening_text', 'opening_text.plain',
    'category', 'category.plain',
    'redirect.title',
    'outgoing_link'
]

# raw results from hive query
CLICK_DATA_TSV = "%s/source_clicks.tsv" % (DATA_DIR)
# source in h5 format, flattened out into single results
CLICK_DATA = "%s/source_clicks.pkl" % (DATA_DIR)
# relevance labels from a DBN
RELEVANCE_DATA = "%s/relevance.pkl" % (DATA_DIR)

# Text and stats from elasticsearch to augment the data
ES_PAGE_DOCS = "%s/es_page_data.pkl" % (DATA_DIR)
ES_PAGE_TERM_VEC = "%s/es_page_term_vec.pkl" % (DATA_DIR)
ES_QUERY_TERM_VEC = "%s/es_query_term_vec.pkl" % (DATA_DIR)

# Above packaged together with 'plain' analyzed fields
ALL_DATA = "%s/all.pkl" % (DATA_DIR)

FEAT_FILE_SUFFIX = ".pkl"

MISSING_VALUE_STRING = "MISSINGVALUE"
MISSING_VALUE_NUMERIC = -1

# Other
TERM_VEC_MAX_NUM_TERMS = 50
RANDOM_SEED = 42
NUM_CORES = 4

DBN_CONFIG = {
    'MAX_ITERATIONS': 40,
    'DEBUG': False,
    'PRETTY_LOG': True,
    'MIN_DOCS_PER_QUERY': 10,
    'MAX_DOCS_PER_QUERY': 10,
    'SERP_SIZE': 10,
    'QUERY_INDEPENDENT_PAGER': False,
    'DEFAULT_REL': 0.5
}
DIRS = [
    DATA_DIR,
    FEAT_DIR,
    LOG_DIR,
    TMP_DIR,
]

os_utils._create_dirs(DIRS)

import pandas as pd

import requests
import grequests
import json
import progressbar

import config
from utils import np_utils, table_utils, es_utils

def exception_handler(req, e):
    raise e

def batch_parse_termvec(batch):
    # Elasticsearch doesn't tell us what the custom doc passed in was, so we need
    # to attach a parse function and manually zip up the results
    def parse(r, *args, **kwargs):
        found = json.loads(r.text)['docs']
        # Can't have a custom return, so piggy-back on the response
        r.data = {}
        for query, doc in zip(list(batch), found):
            r.data[query] = es_utils.parse_termvec(doc)
    return parse


def main():
    queries = table_utils._read(config.CLICK_DATA)[['query', 'norm_query']] \
            .stack().drop_duplicates().reset_index(drop=True).values

    url = config.ES_URL + '/page/_mtermvectors'
    session = requests.Session()
    reqs = []
    for batch in np_utils._split(queries, 100):
        docs = []
        for query in batch:
            doc = {
                'doc': dict(zip(config.ES_TERM_FIELDS, [query] * len(config.ES_TERM_FIELDS))),
                'fields': config.ES_TERM_FIELDS,
                'positions': False,
                'offsets': False,
                'term_statistics': True,
                'filter': {
                    'max_num_terms': 50
                }
            }
            # Custom munging for the redirect.title field
            del doc['doc']['redirect.title']
            doc['doc']['redirect'] = [{'namespace': 0, 'title': query}]
            docs.append(doc)
        # for *extra* funsies, elasticsearch isn't going to give us back the original
        # doc that we searched for, although it could be reconstructed from the tokens
        # under most circumstances.
        reqs.append(grequests.post(url, data=json.dumps({'docs': docs}),
                                   session=session, callback=batch_parse_termvec(batch)))

    data = table_utils._open_shelve_write(config.ES_QUERY_TERM_VEC_SHELVE)
    try:
        with progressbar.ProgressBar(max_value=len(reqs)) as bar:
            for r in bar(grequests.imap(reqs, size=20, exception_handler=exception_handler)):
                for query, vecs in r.data.iteritems():
                    # Can't directly use unicode strings until 3.x, only strings
                    data[query.encode('utf8')] = vecs
    finally:
        data.close()

if __name__ == "__main__":
    main()

import pandas as pd

import requests
import grequests
import json
import progressbar

import config
from utils import np_utils, table_utils

def exception_handler(req, e):
    raise e

def parse_vec(batch):
    def parse(r, *args, **kwargs):
        found = json.loads(r.text)['docs']
        # Can't have a custom return, so piggy-back on the response
        r.data = {}
        for query, doc in zip(list(batch), found):
            result = {}
            for field, data in doc['term_vectors'].iteritems():
                result['query_' + field + '_terms'] = [term for term in data['terms']]
                result['query_' + field + '_score'] = [vec['score'] for _, vec in data['terms'].iteritems()]
                result['query_' + field + '_term_freq'] = [vec['term_freq'] for _, vec in data['terms'].iteritems()]
                result['query_' + field + '_ttf'] = [vec['ttf'] for _, vec in data['terms'].iteritems()]
                result['query_' + field + '_doc_freq'] = [vec['doc_freq'] for _, vec in data['terms'].iteritems()]
            r.data[query] = result
    return parse


def main():
    # returns an ndarray
    queries = table_utils._read(config.CLICK_DATA)['query'].unique()

    url = config.ES_URL + '/page/_mtermvectors'
    session = requests.Session()
    reqs = []
    for batch in np_utils._split(queries, 100):
        docs = []
        for query in batch:
            docs.append({
                'doc': dict(zip(config.ES_TERM_FIELDS, [query] * len(config.ES_TERM_FIELDS))),
                'fields': config.ES_TERM_FIELDS,
                'positions': False,
                'offsets': False,
                'term_statistics': True,
                'filter': {
                    'max_num_terms': 50
                }
            })
        # for *extra* funsies, elasticsearch isn't going to give us back the original
        # doc that we searched for, although it could be reconstructed from the tokens
        # under most circumstances.
        reqs.append(grequests.post(url, data=json.dumps({'docs': docs}),
                                   session=session, callback=parse_vec(batch)))

    data = {}
    with progressbar.ProgressBar(max_value=len(reqs)) as bar:
        for r in bar(grequests.imap(reqs, size=20, exception_handler=exception_handler)):
            data.update(r.data)

    table_utils._write(config.ES_QUERY_TERM_VEC, pd.DataFrame(data).transpose())

if __name__ == "__main__":
    main()

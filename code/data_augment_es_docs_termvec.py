import pandas as pd

import requests
import grequests
import json
import progressbar

import config
from utils import np_utils, table_utils

def exception_handler(req, e):
    raise e

def parse_vec(doc):
    result = {}
    for field, data in doc['term_vectors'].iteritems():
        result['hit_' + field + '_terms'] = [term for term in data['terms']]
        result['hit_' + field + '_score'] = [vec['score'] for _, vec in data['terms'].iteritems()]
        result['hit_' + field + '_term_freq'] = [vec['term_freq'] for _, vec in data['terms'].iteritems()]
        result['hit_' + field + '_ttf'] = [vec['ttf'] for _, vec in data['terms'].iteritems()]
        result['hit_' + field + '_doc_freq'] = [vec['doc_freq'] for _, vec in data['terms'].iteritems()]
    return result


def main():
    # returns an ndarray
    page_ids = table_utils._read(config.CLICK_DATA)['hit_page_id'].unique()

    url = config.ES_URL + '/page/_mtermvectors'
    docs = {}
    i = 0
    with progressbar.ProgressBar(max_value=len(page_ids)) as bar:
        # doing it all in one grequests.imap seems to hold onto a ton
        # of extra memory. On a sample of ~200k page id's i was seeing
        # 14G+, but breaking into smaller pieces brought it down to 7G
        for top_batch in np_utils._split(page_ids, 10000):
            session = requests.Session()
            reqs = []
            for batch in np_utils._split(top_batch, 100):
                data = {
                    'ids': list(batch),
                    'parameters': {
                        'fields': config.ES_TERM_FIELDS,
                        'positions': False,
                        'offsets': False,
                        'term_statistics': True,
                        'filter': {
                            'max_num_terms': config.TERM_VEC_MAX_NUM_TERMS,
                        }
                    }
                }
                reqs.append(grequests.post(url, data=json.dumps(data), session=session))

            for r in grequests.imap(reqs, size=20, exception_handler=exception_handler):
                found = json.loads(r.text)['docs']
                docs.update({int(d['_id']): parse_vec(d) for d in found if d['found']})
                i += len(found)
                bar.update(i)

    table_utils._write(config.ES_PAGE_TERM_VEC, pd.DataFrame(docs).transpose())

if __name__ == "__main__":
    main()

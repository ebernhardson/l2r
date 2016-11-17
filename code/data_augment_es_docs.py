import pandas as pd

import requests
import grequests
import json
import progressbar

import config
from utils import np_utils, table_utils

def exception_handler(req, e):
    raise e

def main():
    # returns an ndarray
    page_ids = table_utils._read(config.CLICK_DATA)['hit_page_id'].unique()

    url = config.ES_URL + '/page/_mget'
    params = {'fields': ','.join(config.ES_DOC_FIELDS)}
    session = requests.Session()
    reqs = []
    for batch in np_utils._split(page_ids, 1000):
        data = json.dumps({'ids': list(batch)})
        reqs.append(grequests.post(url, data=data, params=params, session=session))

    docs = {}
    with progressbar.ProgressBar(max_value=len(reqs)) as bar:
        for r in bar(grequests.imap(reqs, size=20, exception_handler=exception_handler)):
            found = json.loads(r.text)['docs']
            for d in found:
                if d['found']:
                    # ES alwards returns a list, even if there is only one item. For now
                    # we are only ever requesting single-value fields, so take the first
                    docs[int(d['_id'])] = {k: v[0] for k, v in d['fields'].iteritems()}

    table_utils._write(config.ES_PAGE_DOCS, pd.DataFrame(docs).transpose())

if __name__ == "__main__":
    main()

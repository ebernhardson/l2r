import sys

sys.path.append('..')
import config

def parse_termvec(doc):
    result = {}
    for field, data in doc['term_vectors'].iteritems():
        result[field] = {
            'terms': data['terms'].keys(),
        }
        for f in ['score', 'term_freq', 'ttf', 'doc_freq']:
            result[field][f] = [vec[f] for vec in data['terms'].values()]
    for field in config.ES_TERM_FIELDS:
        if field not in result:
            result[field] = {}
            for f in ['terms', 'score', 'term_freq', 'ttf', 'doc_freq']:
                result[field][f] = []
    return result

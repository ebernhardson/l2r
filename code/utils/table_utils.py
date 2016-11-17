import pickle

def _write(fname, data, protocol=2):
    with open(fname, 'wb') as f:
        pickle.dump(data, f, protocol)

def _read(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


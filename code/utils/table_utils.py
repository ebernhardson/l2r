import pickle
import shelve

def _write(fname, data, protocol=2):
    with open(fname, 'wb') as f:
        pickle.dump(data, f, protocol)

def _read(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

def _open_shelve_read(fname):
    return shelve.open(fname, flag='r', protocol=2)


def _open_shelve_write(fname):
    # Shelves are considered write-once, so this truncates
    # any existing database
    return shelve.open(fname, flag='n', protocol=2)

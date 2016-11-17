
def _unigrams(words):
    assert type(words) == list
    return words

def _bigrams(words, join_string):
    assert type(words) == list
    N = len(words)
    if N > 1:
        lst = []
        for i in xrange(N-1):
            lst.append(join_string.join([words[i], words[i+1]]))
    else:
        lst = _unigrams(words)
    return lst

def _trigrams(words, join_string):
    assert type(words) == list
    N = len(words)
    if N > 2:
        lst = []
        for i in xrange(N-2):
            lst.append(join_string.join([words[i], words[i+1], words[i+2]]))
    else:
        lst = _bigrams(words, join_string)
    return lst

def _fourgrams(words, join_string):
    assert type(words) == list
    N = len(words)
    if N > 3:
        lst = []
        for i in xrange(N-3):
            lst.append(join_string.join([words[i], words[i+1], words[i+2], words[i+3]]))
    else:
        lst = _trigrams(words, join_string)
    return lst

def _uniterms(words):
    return _unigrams(words)

def _biterms(words, join_string):
    assert type(words) == list
    N = len(words)
    if N > 1:
        lst = []
        for i in range(N-1):
            for j in range(i+1, N):
                lst.append(join_string.join([words[i], words[j]]))
    else:
        lst = _unigrams(words)
    return lst

def _triterms(words, join_string):
    assert type(words) == list
    N = len(words)
    if N > 2:
        lst = []
        for i in range(N-2):
            for j in range(i+1, N-1):
                for k in range(j+1, N):
                    lst.append(join_string.join([words[i], words[j], words[k]]))
    else:
        lst = _bigram(words, join_string, skip)
    return lst


def _fourterms(words, join_string):
    assert type(words) == list
    N = len(words)
    if N > 3:
        lst = []
        for i in range(N-3):
            for j in range(i+1, N-2):
                for k in range(j+1, N-1):
                    for l in range(k+1, N):
                        lst.append(join_string.join([words[i], words[j], words[k], words[l]]))
    else:
        lst = _triterms(words, join_string, skip)
    return lst

_ngram_str_map = {
    1: "Unigram",
    2: "Bigram",
    3: "Trigram",
    4: "Fourgram",
    5: "Fivegram",
    # unigrams and bigrams
    12: "UBgram",
    #  unigrams, bigrams, trigrams
    123: "UBTgram",
}

def _ngrams(words, ngram, join_string=" "):
    if ngram == 1:
        return _unigrams(words)
    elif ngram == 2:
        return _bigrams(words, join_string)
    elif ngram == 3:
        return _trigrams(words, join_string)
    elif ngram == 4:
        return _fourgrams(words, join_string)
    elif ngram == 12:
        unigram = _unigrams(words)
        bigram = [x for x in _bigrams(words, join_string) if len(x.split(join_string)) == 2]
        return unigram + bigram
    elif ngram == 123:
        unigram = _ungrams(words)
        bigram = [x for x in _bigrams(words, join_string) if len(x.split(join_string)) == 2]
        trigram = [x for x in _trigrams(words, join_string) if len(x.split(join_string)) == 3]
        return unigram + bigram + trigram

_nterm_str_map = {
    1: "Uniterm",
    2: "Biterm",
    3: "Triterm",
    4: "Fourterm",
    5: "Fiveterm",
}

def _nterms(words, nterm, join_string=" "):
    if nterm == 1:
        return _uniterms(words)
    elif nterm == 2:
        return _biterms(words, join_string)
    elif nterm == 3:
        return _triterms(words, join_string)
    elif nterm == 4:
        return _fourterms(words, join_string)

if __name__ == "__main__":
    text = "This is a test str"
    words = text.split(" ")

    assert _ngrams(words, 1) == ["This", "is", "a", "test", "str"]
    assert _ngrams(words, 2) == ["This is", "is a", "a test", "test str"]
    assert _ngrams(words, 3) == ["This is a", "is a test", "a test str"]
    assert _ngrams(words, 4) == ["This is a test", "is a test str"]

    assert _nterms(words, 1) == ["This", "is", "a", "test", "str"]
    assert _nterms(words, 2) == ["This is", "This a", "This test", "This str", "is a", "is test", "is str", "a test", "a str", "test str"]
    assert _nterms(words, 3) == ["This is a", "This is test", "This is str", "This a test", "This a str", "This test str", "is a test", "is a str", "is test str", "a test str"]
    assert _nterms(words, 4) == ["This is a test", "This is a str", "This is test str", "This a test str", "is a test str"]

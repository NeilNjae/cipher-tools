import string
import norms
import random
import collections
import unicodedata
import itertools
from math import log10

def letters(text):
    """Remove all non-alphabetic characters from a text
    >>> letters('The Quick')
    'TheQuick'
    >>> letters('The Quick BROWN fox jumped! over... the (9lazy) DOG')
    'TheQuickBROWNfoxjumpedoverthelazyDOG'
    """
    return ''.join([c for c in text if c in string.ascii_letters])

def unaccent(text):
    """Remove all accents from letters. 
    It does this by converting the unicode string to decomposed compatability
    form, dropping all the combining accents, then re-encoding the bytes.

    >>> unaccent('hello')
    'hello'
    >>> unaccent('HELLO')
    'HELLO'
    >>> unaccent('héllo')
    'hello'
    >>> unaccent('héllö')
    'hello'
    >>> unaccent('HÉLLÖ')
    'HELLO'
    """
    return unicodedata.normalize('NFKD', text).\
        encode('ascii', 'ignore').\
        decode('utf-8')

def sanitise(text):
    """Remove all non-alphabetic characters and convert the text to lowercase
    
    >>> sanitise('The Quick')
    'thequick'
    >>> sanitise('The Quick BROWN fox jumped! over... the (9lazy) DOG')
    'thequickbrownfoxjumpedoverthelazydog'
    >>> sanitise('HÉLLÖ')
    'hello'
    """
    # sanitised = [c.lower() for c in text if c in string.ascii_letters]
    # return ''.join(sanitised)
    return letters(unaccent(text)).lower()


def datafile(name, sep='\t'):
    """Read key,value pairs from file.
    """
    with open(name, 'r') as f:
        for line in f:
            splits = line.split(sep)
            yield [splits[0], int(splits[1])]

english_counts = collections.Counter(dict(datafile('count_1l.txt')))
normalised_english_counts = norms.normalise(english_counts)

english_bigram_counts = collections.Counter(dict(datafile('count_2l.txt')))
normalised_english_bigram_counts = norms.normalise(english_bigram_counts)

english_trigram_counts = collections.Counter(dict(datafile('count_3l.txt')))
normalised_english_trigram_counts = norms.normalise(english_trigram_counts)

with open('words.txt', 'r') as f:
    keywords = [line.rstrip() for line in f]


def weighted_choice(d):
	"""Generate random item from a dictionary of item counts
	"""
	target = random.uniform(0, sum(d.values()))
	cuml = 0.0
	for (l, p) in d.items():
		cuml += p
		if cuml > target:
			return l
	return None

def random_english_letter():
	"""Generate a random letter based on English letter counts
	"""
	return weighted_choice(normalised_english_counts)


def ngrams(text, n):
    """Returns all n-grams of a text
    
    >>> ngrams(sanitise('the quick brown fox'), 2) # doctest: +NORMALIZE_WHITESPACE
    ['th', 'he', 'eq', 'qu', 'ui', 'ic', 'ck', 'kb', 'br', 'ro', 'ow', 'wn', 
     'nf', 'fo', 'ox']
    >>> ngrams(sanitise('the quick brown fox'), 4) # doctest: +NORMALIZE_WHITESPACE
    ['theq', 'hequ', 'equi', 'quic', 'uick', 'ickb', 'ckbr', 'kbro', 'brow', 
     'rown', 'ownf', 'wnfo', 'nfox']
    """
    return [text[i:i+n] for i in range(len(text)-n+1)]


class Pdist(dict):
    """A probability distribution estimated from counts in datafile.
    Values are stored and returned as log probabilities.
    """
    def __init__(self, data=[], estimate_of_missing=None):
        data1, data2 = itertools.tee(data)
        self.total = sum([d[1] for d in data1])
        for key, count in data2:
            self[key] = log10(count / self.total)
        self.estimate_of_missing = estimate_of_missing or (lambda k, N: 1./N)
    def __missing__(self, key):
        return self.estimate_of_missing(key, self.total)

def log_probability_of_unknown_word(key, N):
    """Estimate the probability of an unknown word.
    """
    return -log10(N * 10**((len(key) - 2) * 1.4))

Pw = Pdist(datafile('count_1w.txt'), log_probability_of_unknown_word)
Pl = Pdist(datafile('count_1l.txt'), lambda _k, _N: 0)
P2l = Pdist(datafile('count_2l.txt'), lambda _k, _N: 0)

def Pwords(words): 
    """The Naive Bayes log probability of a sequence of words.
    """
    return sum(Pw[w.lower()] for w in words)

def Pletters(letters):
    """The Naive Bayes log probability of a sequence of letters.
    """
    return sum(Pl[l.lower()] for l in letters)

def Pbigrams(letters):
    """The Naive Bayes log probability of the bigrams formed from a sequence 
    of letters.
    """
    return sum(P2l[p] for p in ngrams(letters, 2))


def cosine_distance_score(text):
    """Finds the dissimilarity of a text to English, using the cosine distance
    of the frequency distribution.

    >>> cosine_distance_score('abcabc') # doctest: +ELLIPSIS
    0.370847405...
    """
    return norms.cosine_distance(english_counts, 
        collections.Counter(sanitise(text)))


if __name__ == "__main__":
    import doctest
    doctest.testmod()

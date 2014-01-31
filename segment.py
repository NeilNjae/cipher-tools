import string
import collections
from math import log10
import itertools
import sys
from functools import lru_cache
sys.setrecursionlimit(1000000)

@lru_cache()
def segment(text):
    """Return a list of words that is the best segmentation of text.
    """
    if not text: return []
    candidates = ([first]+segment(rest) for first,rest in splits(text))
    return max(candidates, key=Pwords)

def splits(text, L=20):
    """Return a list of all possible (first, rest) pairs, len(first)<=L.
    """
    return [(text[:i+1], text[i+1:]) 
            for i in range(min(len(text), L))]

def Pwords(words): 
    """The Naive Bayes log probability of a sequence of words.
    """
    return sum(Pw[w.lower()] for w in words)

class Pdist(dict):
    """A probability distribution estimated from counts in datafile.
    Values are stored and returned as log probabilities.
    """
    def __init__(self, data=[], estimate_of_missing=None):
        data1, data2 = itertools.tee(data)
        self.total = sum([int(d[1]) for d in data1])
        for key, count in data2:
            self[key] = log10(int(count) / self.total)
        self.estimate_of_missing = estimate_of_missing or (lambda k, N: 1./N)
    def __missing__(self, key):
        return self.estimate_of_missing(key, self.total)

def datafile(name, sep='\t'):
    """Read key,value pairs from file.
    """
    with open(name, 'r') as f:
        for line in f:
            yield line.split(sep)

def avoid_long_words(key, N):
    """Estimate the probability of an unknown word.
    """
    return -log10((N * 10**(len(key) - 2)))

Pw  = Pdist(datafile('count_1w.txt'), avoid_long_words)
    

import language_models
import sys
from functools import lru_cache
sys.setrecursionlimit(1000000)

@lru_cache()
def segment(text):
    """Return a list of words that is the best segmentation of text.
    """
    if not text: return []
    candidates = ([first]+segment(rest) for first,rest in splits(text))
    return max(candidates, key=language_models.Pwords)

def splits(text, L=20):
    """Return a list of all possible (first, rest) pairs, len(first)<=L.
    """
    return [(text[:i+1], text[i+1:]) 
            for i in range(min(len(text), L))]


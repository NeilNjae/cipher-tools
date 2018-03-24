import collections
from math import log10


def lp(v1, v2=None, p=2):
    """Find the L_p norm. If passed one vector, find the length of that vector.
    If passed two vectors, find the length of the difference between them.
    """
    if v2:
        vec = {k: abs(v1[k] - v2[k]) for k in (v1.keys() | v2.keys())}
    else:
        vec = v1
    return sum(v ** p for v in vec.values()) ** (1.0 / p)

def l1(v1, v2=None):
    """Finds the distances between two frequency profiles, expressed as 
    dictionaries. Assumes every key in frequencies1 is also in frequencies2

    >>> l1({'a':1, 'b':1, 'c':1}, {'a':1, 'b':1, 'c':1})
    0.0
    >>> l1({'a':2, 'b':2, 'c':2}, {'a':1, 'b':1, 'c':1})
    3.0
    >>> l1(normalise({'a':2, 'b':2, 'c':2}), normalise({'a':1, 'b':1, 'c':1}))
    0.0
    >>> l1({'a':0, 'b':2, 'c':0}, {'a':1, 'b':1, 'c':1})
    3.0
    >>> l1({'a':0, 'b':1}, {'a':1, 'b':1})
    1.0
    """    
    return lp(v1, v2, 1)

def l2(v1, v2=None):
    """Finds the distances between two frequency profiles, expressed as dictionaries.
    Assumes every key in frequencies1 is also in frequencies2
    
    >>> l2({'a':1, 'b':1, 'c':1}, {'a':1, 'b':1, 'c':1})
    0.0
    >>> l2({'a':2, 'b':2, 'c':2}, {'a':1, 'b':1, 'c':1}) # doctest: +ELLIPSIS
    1.73205080...
    >>> l2(normalise({'a':2, 'b':2, 'c':2}), normalise({'a':1, 'b':1, 'c':1}))
    0.0
    >>> l2({'a':0, 'b':2, 'c':0}, {'a':1, 'b':1, 'c':1}) # doctest: +ELLIPSIS
    1.732050807...
    >>> l2(normalise({'a':0, 'b':2, 'c':0}), \
           normalise({'a':1, 'b':1, 'c':1})) # doctest: +ELLIPSIS
    0.81649658...
    >>> l2({'a':0, 'b':1}, {'a':1, 'b':1})
    1.0
    """
    return lp(v1, v2, 2)

def l3(v1, v2=None):
    """Finds the distances between two frequency profiles, expressed as 
    dictionaries. Assumes every key in frequencies1 is also in frequencies2

    >>> l3({'a':1, 'b':1, 'c':1}, {'a':1, 'b':1, 'c':1})
    0.0
    >>> l3({'a':2, 'b':2, 'c':2}, {'a':1, 'b':1, 'c':1}) # doctest: +ELLIPSIS
    1.44224957...
    >>> l3({'a':0, 'b':2, 'c':0}, {'a':1, 'b':1, 'c':1}) # doctest: +ELLIPSIS
    1.4422495703...
    >>> l3(normalise({'a':0, 'b':2, 'c':0}), \
           normalise({'a':1, 'b':1, 'c':1})) # doctest: +ELLIPSIS
    0.718144896...
    >>> l3({'a':0, 'b':1}, {'a':1, 'b':1})
    1.0
    >>> l3(normalise({'a':0, 'b':1}), normalise({'a':1, 'b':1})) # doctest: +ELLIPSIS
    0.6299605249...
    """
    return lp(v1, v2, 3)

def linf(v1, v2=None):    
    if v2:
        vec = {k: abs(v1[k] - v2[k]) for k in (v1.keys() | v2.keys())}
    else:
        vec = v1
    return max(v for v in vec.values())


def scale(frequencies, norm=l2):
    length = norm(frequencies)
    return collections.defaultdict(int, 
        {k: v / length for k, v in frequencies.items()})

def l2_scale(f):
    """Scale a set of frequencies so they have a unit euclidean length
    
    >>> sorted(euclidean_scale({1: 1, 2: 0}).items())
    [(1, 1.0), (2, 0.0)]
    >>> sorted(euclidean_scale({1: 1, 2: 1}).items()) # doctest: +ELLIPSIS
    [(1, 0.7071067...), (2, 0.7071067...)]
    >>> sorted(euclidean_scale({1: 1, 2: 1, 3: 1}).items()) # doctest: +ELLIPSIS
    [(1, 0.577350...), (2, 0.577350...), (3, 0.577350...)]
    >>> sorted(euclidean_scale({1: 1, 2: 2, 3: 1}).items()) # doctest: +ELLIPSIS
    [(1, 0.408248...), (2, 0.81649658...), (3, 0.408248...)]
    """
    return scale(f, l2)

def l1_scale(f):
    """Scale a set of frequencies so they sum to one
    
    >>> sorted(normalise({1: 1, 2: 0}).items())
    [(1, 1.0), (2, 0.0)]
    >>> sorted(normalise({1: 1, 2: 1}).items())
    [(1, 0.5), (2, 0.5)]
    >>> sorted(normalise({1: 1, 2: 1, 3: 1}).items()) # doctest: +ELLIPSIS
    [(1, 0.333...), (2, 0.333...), (3, 0.333...)]
    >>> sorted(normalise({1: 1, 2: 2, 3: 1}).items())
    [(1, 0.25), (2, 0.5), (3, 0.25)]
    """    
    return scale(f, l1)

normalise = l1_scale
euclidean_distance = l2
euclidean_scale = l2_scale


def geometric_mean(frequencies1, frequencies2):
    """Finds the geometric mean of the absolute differences between two frequency profiles, 
    expressed as dictionaries.
    Assumes every key in frequencies1 is also in frequencies2
    
    >>> geometric_mean({'a':2, 'b':2, 'c':2}, {'a':1, 'b':1, 'c':1})
    1.0
    >>> geometric_mean({'a':2, 'b':2, 'c':2}, {'a':1, 'b':1, 'c':1})
    1.0
    >>> geometric_mean({'a':2, 'b':2, 'c':2}, {'a':1, 'b':5, 'c':1})
    3.0
    >>> geometric_mean(normalise({'a':2, 'b':2, 'c':2}), \
                       normalise({'a':1, 'b':5, 'c':1})) # doctest: +ELLIPSIS
    0.01382140...
    >>> geometric_mean(normalise({'a':2, 'b':2, 'c':2}), \
                       normalise({'a':1, 'b':1, 'c':1})) # doctest: +ELLIPSIS
    0.0
    >>> geometric_mean(normalise({'a':2, 'b':2, 'c':2}), \
                       normalise({'a':1, 'b':1, 'c':0})) # doctest: +ELLIPSIS
    0.009259259...
    """
    total = 1.0
    for k in frequencies1:
        total *= abs(frequencies1[k] - frequencies2[k])
    return total

def harmonic_mean(frequencies1, frequencies2):
    """Finds the harmonic mean of the absolute differences between two frequency profiles, 
    expressed as dictionaries.
    Assumes every key in frequencies1 is also in frequencies2

    >>> harmonic_mean({'a':2, 'b':2, 'c':2}, {'a':1, 'b':1, 'c':1})
    1.0
    >>> harmonic_mean({'a':2, 'b':2, 'c':2}, {'a':1, 'b':1, 'c':1})
    1.0
    >>> harmonic_mean({'a':2, 'b':2, 'c':2}, {'a':1, 'b':5, 'c':1}) # doctest: +ELLIPSIS
    1.285714285...
    >>> harmonic_mean(normalise({'a':2, 'b':2, 'c':2}), \
                      normalise({'a':1, 'b':5, 'c':1})) # doctest: +ELLIPSIS
    0.228571428571...
    >>> harmonic_mean(normalise({'a':2, 'b':2, 'c':2}), \
                      normalise({'a':1, 'b':1, 'c':1})) # doctest: +ELLIPSIS
    0.0
    >>> harmonic_mean(normalise({'a':2, 'b':2, 'c':2}), \
                      normalise({'a':1, 'b':1, 'c':0})) # doctest: +ELLIPSIS
    0.2
    """
    total = 0.0
    for k in frequencies1:
        if abs(frequencies1[k] - frequencies2[k]) == 0:
            return 0.0
        total += 1.0 / abs(frequencies1[k] - frequencies2[k])
    return len(frequencies1) / total


def cosine_similarity(frequencies1, frequencies2):
    """Finds the distances between two frequency profiles, expressed as dictionaries.
    Assumes every key in frequencies1 is also in frequencies2

    >>> cosine_similarity({'a':1, 'b':1, 'c':1}, {'a':1, 'b':1, 'c':1}) # doctest: +ELLIPSIS
    1.0000000000...
    >>> cosine_similarity({'a':2, 'b':2, 'c':2}, {'a':1, 'b':1, 'c':1}) # doctest: +ELLIPSIS
    1.0000000000...
    >>> cosine_similarity({'a':0, 'b':2, 'c':0}, {'a':1, 'b':1, 'c':1}) # doctest: +ELLIPSIS
    0.5773502691...
    >>> cosine_similarity({'a':0, 'b':1}, {'a':1, 'b':1}) # doctest: +ELLIPSIS
    0.7071067811...
    """
    numerator = 0
    length1 = 0
    length2 = 0
    for k in frequencies1:
        numerator += frequencies1[k] * frequencies2[k]
        length1 += frequencies1[k]**2
    for k in frequencies2:
        length2 += frequencies2[k]**2
    return numerator / (length1 ** 0.5 * length2 ** 0.5)



if __name__ == "__main__":
    import doctest
    doctest.testmod()

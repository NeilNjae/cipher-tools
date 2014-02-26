import collections
from math import log10

def normalise(frequencies):
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
    length = sum([f for f in frequencies.values()])
    return collections.defaultdict(int, ((k, v / length) 
        for (k, v) in frequencies.items()))

def euclidean_scale(frequencies):
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
    length = sum([f ** 2 for f in frequencies.values()]) ** 0.5
    return collections.defaultdict(int, ((k, v / length) 
        for (k, v) in frequencies.items()))

def identity_scale(frequencies):
    return frequencies
        

def l2(frequencies1, frequencies2):
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
    total = 0
    for k in frequencies1:
        total += (frequencies1[k] - frequencies2[k]) ** 2
    return total ** 0.5
euclidean_distance = l2

def l1(frequencies1, frequencies2):
    """Finds the distances between two frequency profiles, expressed as 
    dictionaries. Assumes every key in frequencies1 is also in frequencies2

    >>> l1({'a':1, 'b':1, 'c':1}, {'a':1, 'b':1, 'c':1})
    0
    >>> l1({'a':2, 'b':2, 'c':2}, {'a':1, 'b':1, 'c':1})
    3
    >>> l1(normalise({'a':2, 'b':2, 'c':2}), normalise({'a':1, 'b':1, 'c':1}))
    0.0
    >>> l1({'a':0, 'b':2, 'c':0}, {'a':1, 'b':1, 'c':1})
    3
    >>> l1({'a':0, 'b':1}, {'a':1, 'b':1})
    1
    """
    total = 0
    for k in frequencies1:
        total += abs(frequencies1[k] - frequencies2[k])
    return total

def l3(frequencies1, frequencies2):
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
    total = 0
    for k in frequencies1:
        total += abs(frequencies1[k] - frequencies2[k]) ** 3
    return total ** (1/3)

def geometric_mean(frequencies1, frequencies2):
    """Finds the geometric mean of the absolute differences between two frequency profiles, 
    expressed as dictionaries.
    Assumes every key in frequencies1 is also in frequencies2
    
    >>> geometric_mean({'a':2, 'b':2, 'c':2}, {'a':1, 'b':1, 'c':1})
    1
    >>> geometric_mean({'a':2, 'b':2, 'c':2}, {'a':1, 'b':1, 'c':1})
    1
    >>> geometric_mean({'a':2, 'b':2, 'c':2}, {'a':1, 'b':5, 'c':1})
    3
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
    total = 1
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
    0
    >>> harmonic_mean(normalise({'a':2, 'b':2, 'c':2}), \
                      normalise({'a':1, 'b':1, 'c':0})) # doctest: +ELLIPSIS
    0.2
    """
    total = 0
    for k in frequencies1:
        if abs(frequencies1[k] - frequencies2[k]) == 0:
            return 0
        total += 1 / abs(frequencies1[k] - frequencies2[k])
    return len(frequencies1) / total


def cosine_distance(frequencies1, frequencies2):
    """Finds the distances between two frequency profiles, expressed as dictionaries.
    Assumes every key in frequencies1 is also in frequencies2

    >>> cosine_distance({'a':1, 'b':1, 'c':1}, {'a':1, 'b':1, 'c':1}) # doctest: +ELLIPSIS
    1.0000000000...
    >>> cosine_distance({'a':2, 'b':2, 'c':2}, {'a':1, 'b':1, 'c':1}) # doctest: +ELLIPSIS
    1.0000000000...
    >>> cosine_distance({'a':0, 'b':2, 'c':0}, {'a':1, 'b':1, 'c':1}) # doctest: +ELLIPSIS
    0.5773502691...
    >>> cosine_distance({'a':0, 'b':1}, {'a':1, 'b':1}) # doctest: +ELLIPSIS
    0.7071067811...
    """
    numerator = 0
    length1 = 0
    length2 = 0
    for k in frequencies1:
        numerator += frequencies1[k] * frequencies2[k]
        length1 += frequencies1[k]**2
    for k in frequencies2.keys():
        length2 += frequencies2[k]
    return numerator / (length1 ** 0.5 * length2 ** 0.5)



if __name__ == "__main__":
    import doctest
    doctest.testmod()

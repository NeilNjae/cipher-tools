import collections

def normalise(frequencies):
    """Scale a set of frequenies so they have a unit euclidean length
    
    >>> sorted(normalise({1: 1, 2: 0}).items())
    [(1, 1.0), (2, 0.0)]
    >>> sorted(normalise({1: 1, 2: 1}).items())
    [(1, 0.7071067811865475), (2, 0.7071067811865475)]
    >>> sorted(normalise({1: 1, 2: 1, 3: 1}).items())
    [(1, 0.5773502691896258), (2, 0.5773502691896258), (3, 0.5773502691896258)]
    >>> sorted(normalise({1: 1, 2: 2, 3: 1}).items())
    [(1, 0.4082482904638631), (2, 0.8164965809277261), (3, 0.4082482904638631)]
   """
    length = sum([f ** 2 for f in frequencies.values()]) ** 0.5
    return collections.defaultdict(int, ((k, v / length) for (k, v) in frequencies.items()))

def scale(frequencies):
    """Scale a set of frequencies so the largest is 1
    
    >>> sorted(scale({1: 1, 2: 0}).items())
    [(1, 1.0), (2, 0.0)]
    >>> sorted(scale({1: 1, 2: 1}).items())
    [(1, 1.0), (2, 1.0)]
    >>> sorted(scale({1: 1, 2: 1, 3: 1}).items())
    [(1, 1.0), (2, 1.0), (3, 1.0)]
    >>> sorted(scale({1: 1, 2: 2, 3: 1}).items())
    [(1, 0.5), (2, 1.0), (3, 0.5)]
    """
    largest = max(frequencies.values())
    return collections.defaultdict(int, ((k, v / largest) for (k, v) in frequencies.items()))
    

def l2(frequencies1, frequencies2):
    """Finds the distances between two frequency profiles, expressed as dictionaries.
    Assumes every key in frequencies1 is also in frequencies2
    
    >>> l2({'a':1, 'b':1, 'c':1}, {'a':1, 'b':1, 'c':1})
    0.0
    >>> l2({'a':2, 'b':2, 'c':2}, {'a':1, 'b':1, 'c':1})
    1.7320508075688772
    >>> l2(normalise({'a':2, 'b':2, 'c':2}), normalise({'a':1, 'b':1, 'c':1}))
    0.0
    >>> l2({'a':0, 'b':2, 'c':0}, {'a':1, 'b':1, 'c':1})
    1.7320508075688772
    >>> l2(normalise({'a':0, 'b':2, 'c':0}), normalise({'a':1, 'b':1, 'c':1}))
    0.9194016867619662
    >>> l2({'a':0, 'b':1}, {'a':1, 'b':1})
    1.0
    """
    total = 0
    for k in frequencies1.keys():
        total += (frequencies1[k] - frequencies2[k]) ** 2
    return total ** 0.5
euclidean_distance = l2

def l1(frequencies1, frequencies2):
    """Finds the distances between two frequency profiles, expressed as dictionaries.
    Assumes every key in frequencies1 is also in frequencies2

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
    for k in frequencies1.keys():
        total += abs(frequencies1[k] - frequencies2[k])
    return total

def l3(frequencies1, frequencies2):
    """Finds the distances between two frequency profiles, expressed as dictionaries.
    Assumes every key in frequencies1 is also in frequencies2

    >>> l3({'a':1, 'b':1, 'c':1}, {'a':1, 'b':1, 'c':1})
    0.0
    >>> l3({'a':2, 'b':2, 'c':2}, {'a':1, 'b':1, 'c':1})
    1.4422495703074083
    >>> l3({'a':0, 'b':2, 'c':0}, {'a':1, 'b':1, 'c':1})
    1.4422495703074083
    >>> l3(normalise({'a':0, 'b':2, 'c':0}), normalise({'a':1, 'b':1, 'c':1}))
    0.7721675487598008
    >>> l3({'a':0, 'b':1}, {'a':1, 'b':1})
    1.0
    >>> l3(normalise({'a':0, 'b':1}), normalise({'a':1, 'b':1}))
    0.7234757712960591
    """
    total = 0
    for k in frequencies1.keys():
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
    >>> geometric_mean(normalise({'a':2, 'b':2, 'c':2}), normalise({'a':1, 'b':5, 'c':1}))
    0.057022248808851934
    >>> geometric_mean(normalise({'a':2, 'b':2, 'c':2}), normalise({'a':1, 'b':1, 'c':1}))
    0.0
    >>> geometric_mean(normalise({'a':2, 'b':2, 'c':2}), normalise({'a':1, 'b':1, 'c':0}))
    0.009720703533656434
    """
    total = 1
    for k in frequencies1.keys():
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
    >>> harmonic_mean({'a':2, 'b':2, 'c':2}, {'a':1, 'b':5, 'c':1})
    1.2857142857142858
    >>> harmonic_mean(normalise({'a':2, 'b':2, 'c':2}), normalise({'a':1, 'b':5, 'c':1}))
    0.3849001794597505
    >>> harmonic_mean(normalise({'a':2, 'b':2, 'c':2}), normalise({'a':1, 'b':1, 'c':1}))
    0
    >>> harmonic_mean(normalise({'a':2, 'b':2, 'c':2}), normalise({'a':1, 'b':1, 'c':0}))
    0.17497266360581604
    """
    total = 0
    for k in frequencies1.keys():
        if abs(frequencies1[k] - frequencies2[k]) == 0:
            return 0
        total += 1 / abs(frequencies1[k] - frequencies2[k])
    return len(frequencies1) / total


def cosine_distance(frequencies1, frequencies2):
    """Finds the distances between two frequency profiles, expressed as dictionaries.
    Assumes every key in frequencies1 is also in frequencies2

    >>> cosine_distance({'a':1, 'b':1, 'c':1}, {'a':1, 'b':1, 'c':1})
    -2.220446049250313e-16
    >>> cosine_distance({'a':2, 'b':2, 'c':2}, {'a':1, 'b':1, 'c':1})
    -2.220446049250313e-16
    >>> cosine_distance({'a':0, 'b':2, 'c':0}, {'a':1, 'b':1, 'c':1})
    0.42264973081037416
    >>> cosine_distance({'a':0, 'b':1}, {'a':1, 'b':1})
    0.29289321881345254
    """
    numerator = 0
    length1 = 0
    length2 = 0
    for k in frequencies1.keys():
        numerator += frequencies1[k] * frequencies2[k]
        length1 += frequencies1[k]**2
    for k in frequencies2.keys():
        length2 += frequencies2[k]
    return 1 - (numerator / (length1 ** 0.5 * length2 ** 0.5))


if __name__ == "__main__":
    import doctest
    doctest.testmod()

"""A set of functions to break the ciphers give in ciphers.py.
"""

import string
import collections
import norms
import logging
import random
import math
from itertools import starmap
from segment import segment
from multiprocessing import Pool

import matplotlib.pyplot as plt


from cipher import *
from language_models import *

# To time a run:
#
# import timeit
# c5a = open('2012/5a.ciphertext', 'r').read()
# timeit.timeit('keyword_break(c5a)', setup='gc.enable() ; from __main__ import c5a ; from cipher import keyword_break', number=1)
# timeit.repeat('keyword_break_mp(c5a, chunksize=500)', setup='gc.enable() ; from __main__ import c5a ; from cipher import keyword_break_mp', repeat=5, number=1)





def amsco_break(message, translist=transpositions, patterns = [(1, 2), (2, 1)],
                                  fillstyles = [AmscoFillStyle.continuous, 
                                                AmscoFillStyle.same_each_row, 
                                                AmscoFillStyle.reverse_each_row],
                                  fitness=Pbigrams, 
                                  chunksize=500):
    """Breaks an AMSCO transposition cipher using a dictionary and
    n-gram frequency analysis

    >>> amsco_break(amsco_transposition_encipher(sanitise( \
            "It is a truth universally acknowledged, that a single man in \
             possession of a good fortune, must be in want of a wife. However \
             little known the feelings or views of such a man may be on his \
             first entering a neighbourhood, this truth is so well fixed in \
             the minds of the surrounding families, that he is considered the \
             rightful property of some one or other of their daughters."), \
        'encipher'), \
        translist={(2, 0, 5, 3, 1, 4, 6): ['encipher'], \
                   (5, 0, 6, 1, 3, 4, 2): ['fourteen'], \
                   (6, 1, 0, 4, 5, 3, 2): ['keyword']}, \
        patterns=[(1, 2)]) # doctest: +ELLIPSIS
    (((2, 0, 5, 3, 1, 4, 6), (1, 2), <AmscoFillStyle.continuous: 1>), -709.4646722...)
    >>> amsco_break(amsco_transposition_encipher(sanitise( \
            "It is a truth universally acknowledged, that a single man in \
             possession of a good fortune, must be in want of a wife. However \
             little known the feelings or views of such a man may be on his \
             first entering a neighbourhood, this truth is so well fixed in \
             the minds of the surrounding families, that he is considered the \
             rightful property of some one or other of their daughters."), \
        'encipher', fillpattern=(2, 1)), \
        translist={(2, 0, 5, 3, 1, 4, 6): ['encipher'], \
                   (5, 0, 6, 1, 3, 4, 2): ['fourteen'], \
                   (6, 1, 0, 4, 5, 3, 2): ['keyword']}, \
        patterns=[(1, 2), (2, 1)], fitness=Ptrigrams) # doctest: +ELLIPSIS
    (((2, 0, 5, 3, 1, 4, 6), (2, 1), <AmscoFillStyle.continuous: 1>), -997.0129085...)
    """
    with Pool() as pool:
        helper_args = [(message, trans, pattern, fillstyle, fitness)
                       for trans in translist
                       for pattern in patterns
                       for fillstyle in fillstyles]
        # Gotcha: the helper function here needs to be defined at the top level
        #   (limitation of Pool.starmap)
        breaks = pool.starmap(amsco_break_worker, helper_args, chunksize) 
        return max(breaks, key=lambda k: k[1])

def amsco_break_worker(message, transposition,
        pattern, fillstyle, fitness):
    plaintext = amsco_transposition_decipher(message, transposition,
        fillpattern=pattern, fillstyle=fillstyle)
    fit = fitness(sanitise(plaintext))
    logger.debug('AMSCO transposition break attempt using key {0} and pattern'
                         '{1} ({2}) gives fit of {3} and decrypt starting: '
                         '{4}'.format(
                             transposition, pattern, fillstyle, fit, 
                             sanitise(plaintext)[:50]))
    return (transposition, pattern, fillstyle), fit


def hill_break(message, matrix_size=2, fitness=Pletters, 
    number_of_solutions=1, chunksize=500):

    all_matrices = [np.matrix(list(m)) 
        for m in itertools.product([list(r) 
            for r in itertools.product(range(26), repeat=matrix_size)], 
        repeat=matrix_size)]
    valid_matrices = [m for m, d in 
        zip(all_matrices, (int(round(linalg.det(m))) for m in all_matrices))
                  if d != 0
                  if d % 2 != 0
                  if d % 13 != 0 ]
    with Pool() as pool:
        helper_args = [(message, matrix, fitness)
                       for matrix in valid_matrices]
        # Gotcha: the helper function here needs to be defined at the top level
        #   (limitation of Pool.starmap)
        breaks = pool.starmap(hill_break_worker, helper_args, chunksize)
        if number_of_solutions == 1:
            return max(breaks, key=lambda k: k[1])
        else:
            return sorted(breaks, key=lambda k: k[1], reverse=True)[:number_of_solutions]

def hill_break_worker(message, matrix, fitness):
    plaintext = hill_decipher(matrix, message)
    fit = fitness(plaintext)
    logger.debug('Hill cipher break attempt using key {0} gives fit of '
                 '{1} and decrypt starting: {2}'.format(matrix, 
                     fit, sanitise(plaintext)[:50]))
    return matrix, fit

def bifid_break_mp(message, wordlist=keywords, fitness=Pletters, max_period=10,
                     number_of_solutions=1, chunksize=500):
    """Breaks a keyword substitution cipher using a dictionary and
    frequency analysis

    >>> bifid_break_mp(bifid_encipher('this is a test message for the ' \
          'keyword decipherment', 'elephant', wrap_alphabet=KeywordWrapAlphabet.from_last), \
          wordlist=['cat', 'elephant', 'kangaroo']) # doctest: +ELLIPSIS
    (('elephant', <KeywordWrapAlphabet.from_last: 2>, 0), -52.834575011...)
    >>> bifid_break_mp(bifid_encipher('this is a test message for the ' \
          'keyword decipherment', 'elephant', wrap_alphabet=KeywordWrapAlphabet.from_last), \
          wordlist=['cat', 'elephant', 'kangaroo'], \
          number_of_solutions=2) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    [(('elephant', <KeywordWrapAlphabet.from_last: 2>, 0), -52.834575011...), 
    (('elephant', <KeywordWrapAlphabet.from_largest: 3>, 0), -52.834575011...)]
    """
    with Pool() as pool:
        helper_args = [(message, word, wrap, period, fitness)
                       for word in wordlist
                       for wrap in KeywordWrapAlphabet
                       for period in range(max_period+1)]
        # Gotcha: the helper function here needs to be defined at the top level
        #   (limitation of Pool.starmap)
        breaks = pool.starmap(bifid_break_worker, helper_args, chunksize)
        if number_of_solutions == 1:
            return max(breaks, key=lambda k: k[1])
        else:
            return sorted(breaks, key=lambda k: k[1], reverse=True)[:number_of_solutions]

def bifid_break_worker(message, keyword, wrap_alphabet, period, fitness):
    plaintext = bifid_decipher(message, keyword, wrap_alphabet, period=period)
    fit = fitness(plaintext)
    logger.debug('Keyword break attempt using key {0} (wrap={1}) gives fit of '
                 '{2} and decrypt starting: {3}'.format(keyword, 
                     wrap_alphabet, fit, sanitise(plaintext)[:50]))
    return (keyword, wrap_alphabet, period), fit


def autokey_sa_break( message
                    , min_keylength=2
                    , max_keylength=20
                    , workers=10
                    , initial_temperature=200
                    , max_iterations=20000
                    , fitness=Pletters
                    , chunksize=1
                    , result_count=1
                    ):
    """Break an autokey cipher by simulated annealing
    """
    worker_args = []
    ciphertext = sanitise(message)
    for keylength in range(min_keylength, max_keylength+1):
        for i in range(workers):
            key = cat(random.choice(string.ascii_lowercase) for _ in range(keylength))
            worker_args.append((ciphertext, key, 
                            initial_temperature, max_iterations, fitness))
            
    with Pool() as pool:
        breaks = pool.starmap(autokey_sa_break_worker,
                              worker_args, chunksize)
    if result_count <= 1:
        return max(breaks, key=lambda k: k[1])
    else:
        return sorted(set(breaks), key=lambda k: k[1], reverse=True)[:result_count]


def autokey_sa_break_worker(message, key, 
                                     t0, max_iterations, fitness):
   
    temperature = t0

    dt = t0 / (0.9 * max_iterations)
    
    plaintext = autokey_decipher(message, key)
    current_fitness = fitness(plaintext)
    current_key = key

    best_key = current_key
    best_fitness = current_fitness
    best_plaintext = plaintext
    
    # print('starting for', max_iterations)
    for i in range(max_iterations):
        swap_pos = random.randrange(len(current_key))
        swap_char = random.choice(string.ascii_lowercase)
        
        new_key = current_key[:swap_pos] + swap_char + current_key[swap_pos+1:]
        
        plaintext = autokey_decipher(message, new_key)
        new_fitness = fitness(plaintext)
        try:
            sa_chance = math.exp((new_fitness - current_fitness) / temperature)
        except (OverflowError, ZeroDivisionError):
            # print('exception triggered: new_fit {}, current_fit {}, temp {}'.format(new_fitness, current_fitness, temperature))
            sa_chance = 0
        if (new_fitness > current_fitness or random.random() < sa_chance):
            # logger.debug('Simulated annealing: iteration {}, temperature {}, '
            #     'current alphabet {}, current_fitness {}, '
            #     'best_plaintext {}'.format(i, temperature, current_alphabet, 
            #     current_fitness, best_plaintext[:50]))

            # logger.debug('new_fit {}, current_fit {}, temp {}, sa_chance {}'.format(new_fitness, current_fitness, temperature, sa_chance))
#             print(new_fitness, new_key, plaintext[:100])
            current_fitness = new_fitness
            current_key = new_key
            
        if current_fitness > best_fitness:
            best_key = current_key
            best_fitness = current_fitness
            best_plaintext = plaintext
        if i % 500 == 0:
            logger.debug('Simulated annealing: iteration {}, temperature {}, '
                'current key {}, current_fitness {}, '
                'best_plaintext {}'.format(i, temperature, current_key, 
                current_fitness, plaintext[:50]))
        temperature = max(temperature - dt, 0.001)
        
#     print(best_key, best_fitness, best_plaintext[:70])
    return best_key, best_fitness # current_alphabet, current_fitness


def pocket_enigma_break_by_crib(message, wheel_spec, crib, crib_position):
    """Break a pocket enigma using a crib (some plaintext that's expected to
    be in a certain position). Returns a list of possible starting wheel
    positions that could produce the crib.

    >>> pocket_enigma_break_by_crib('kzpjlzmoga', 1, 'h', 0)
    ['a', 'f', 'q']
    >>> pocket_enigma_break_by_crib('kzpjlzmoga', 1, 'he', 0)
    ['a']
    >>> pocket_enigma_break_by_crib('kzpjlzmoga', 1, 'll', 2)
    ['a']
    >>> pocket_enigma_break_by_crib('kzpjlzmoga', 1, 'l', 2)
    ['a']
    >>> pocket_enigma_break_by_crib('kzpjlzmoga', 1, 'l', 3)
    ['a', 'j', 'n']
    >>> pocket_enigma_break_by_crib('aaaaa', 1, 'l', 3)
    []
    """
    pe = PocketEnigma(wheel=wheel_spec)
    possible_positions = []
    for p in string.ascii_lowercase:
        pe.set_position(p)
        plaintext = pe.decipher(message)
        if plaintext[crib_position:crib_position+len(crib)] == crib:
            possible_positions += [p]
    return possible_positions


def plot_frequency_histogram(freqs, sort_key=None):
    x = range(len(freqs))
    y = [freqs[l] for l in sorted(freqs, key=sort_key)]
    f = plt.figure()
    ax = f.add_axes([0.1, 0.1, 0.9, 0.9])
    ax.bar(x, y, align='center')
    ax.set_xticks(x)
    ax.set_xticklabels(sorted(freqs, key=sort_key))
    f.show()


if __name__ == "__main__":
    import doctest
    doctest.testmod()

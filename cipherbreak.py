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

logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler('cipher.log'))
logger.setLevel(logging.WARNING)
#logger.setLevel(logging.INFO)
#logger.setLevel(logging.DEBUG)

from cipher import *
from language_models import *

# To time a run:
#
# import timeit
# c5a = open('2012/5a.ciphertext', 'r').read()
# timeit.timeit('keyword_break(c5a)', setup='gc.enable() ; from __main__ import c5a ; from cipher import keyword_break', number=1)
# timeit.repeat('keyword_break_mp(c5a, chunksize=500)', setup='gc.enable() ; from __main__ import c5a ; from cipher import keyword_break_mp', repeat=5, number=1)

transpositions = collections.defaultdict(list)
for word in keywords:
    transpositions[transpositions_of(word)] += [word]

def frequencies(text):
    """Count the number of occurrences of each character in text

    >>> sorted(frequencies('abcdefabc').items())
    [('a', 2), ('b', 2), ('c', 2), ('d', 1), ('e', 1), ('f', 1)]
    >>> sorted(frequencies('the quick brown fox jumped over the lazy ' \
         'dog').items()) # doctest: +NORMALIZE_WHITESPACE
    [(' ', 8), ('a', 1), ('b', 1), ('c', 1), ('d', 2), ('e', 4), ('f', 1),
     ('g', 1), ('h', 2), ('i', 1), ('j', 1), ('k', 1), ('l', 1), ('m', 1),
     ('n', 1), ('o', 4), ('p', 1), ('q', 1), ('r', 2), ('t', 2), ('u', 2),
     ('v', 1), ('w', 1), ('x', 1), ('y', 1), ('z', 1)]
    >>> sorted(frequencies('The Quick BROWN fox jumped! over... the ' \
         '(9lazy) DOG').items()) # doctest: +NORMALIZE_WHITESPACE
    [(' ', 8), ('!', 1), ('(', 1), (')', 1), ('.', 3), ('9', 1), ('B', 1),
     ('D', 1), ('G', 1), ('N', 1), ('O', 2), ('Q', 1), ('R', 1), ('T', 1),
     ('W', 1), ('a', 1), ('c', 1), ('d', 1), ('e', 4), ('f', 1), ('h', 2),
     ('i', 1), ('j', 1), ('k', 1), ('l', 1), ('m', 1), ('o', 2), ('p', 1),
     ('r', 1), ('t', 1), ('u', 2), ('v', 1), ('x', 1), ('y', 1), ('z', 1)]
    >>> sorted(frequencies(sanitise('The Quick BROWN fox jumped! over... '\
         'the (9lazy) DOG')).items()) # doctest: +NORMALIZE_WHITESPACE
    [('a', 1), ('b', 1), ('c', 1), ('d', 2), ('e', 4), ('f', 1), ('g', 1),
     ('h', 2), ('i', 1), ('j', 1), ('k', 1), ('l', 1), ('m', 1), ('n', 1),
     ('o', 4), ('p', 1), ('q', 1), ('r', 2), ('t', 2), ('u', 2), ('v', 1),
     ('w', 1), ('x', 1), ('y', 1), ('z', 1)]
    >>> frequencies('abcdefabcdef')['x']
    0
    """
    return collections.Counter(c for c in text)


def caesar_break(message, fitness=Pletters):
    """Breaks a Caesar cipher using frequency analysis

    >>> caesar_break('ibxcsyorsaqcheyklxivoexlevmrimwxsfiqevvmihrsasrxliwyrh' \
          'ecjsppsamrkwleppfmergefifvmhixscsymjcsyqeoixlm') # doctest: +ELLIPSIS
    (4, -130.849989015...)
    >>> caesar_break('wxwmaxdgheetgwuxztgptedbgznitgwwhpguxyhkxbmhvvtlbhgtee' \
          'raxlmhiixweblmxgxwmhmaxybkbgztgwztsxwbgmxgmert') # doctest: +ELLIPSIS
    (19, -128.82410410...)
    >>> caesar_break('yltbbqnqnzvguvaxurorgenafsbezqvagbnornfgsbevpnaabjurer' \
          'svaquvzyvxrnznazlybequrvfohgriraabjtbaruraprur') # doctest: +ELLIPSIS
    (13, -126.25403935...)
    """
    sanitised_message = sanitise(message)
    best_shift = 0
    best_fit = float('-inf')
    for shift in range(26):
        plaintext = caesar_decipher(sanitised_message, shift)
        fit = fitness(plaintext)
        logger.debug('Caesar break attempt using key {0} gives fit of {1} '
                     'and decrypt starting: {2}'.format(shift, fit,
                                                        plaintext[:50]))
        if fit > best_fit:
            best_fit = fit
            best_shift = shift
    logger.info('Caesar break best fit: key {0} gives fit of {1} and '
                'decrypt starting: {2}'.format(best_shift, best_fit, 
                    caesar_decipher(sanitised_message, best_shift)[:50]))
    return best_shift, best_fit

def affine_break(message, fitness=Pletters):
    """Breaks an affine cipher using frequency analysis

    >>> affine_break('lmyfu bkuusd dyfaxw claol psfaom jfasd snsfg jfaoe ls ' \
          'omytd jlaxe mh jm bfmibj umis hfsul axubafkjamx. ls kffkxwsd jls ' \
          'ofgbjmwfkiu olfmxmtmwaokttg jlsx ls kffkxwsd jlsi zg tsxwjl. jlsx ' \
          'ls umfjsd jlsi zg hfsqysxog. ls dmmdtsd mx jls bats mh bkbsf. ls ' \
          'bfmctsd kfmyxd jls lyj, mztanamyu xmc jm clm cku tmmeaxw kj lai ' \
          'kxd clm ckuxj.') # doctest: +ELLIPSIS
    ((15, 22, True), -340.601181913...)
    """
    sanitised_message = sanitise(message)
    best_multiplier = 0
    best_adder = 0
    best_one_based = True
    best_fit = float("-inf")
    for one_based in [True, False]:
        for multiplier in [x for x in range(1, 26, 2) if x != 13]:
            for adder in range(26):
                plaintext = affine_decipher(sanitised_message,
                                            multiplier, adder, one_based)
                fit = fitness(plaintext)
                logger.debug('Affine break attempt using key {0}x+{1} ({2}) '
                             'gives fit of {3} and decrypt starting: {4}'.
                             format(multiplier, adder, one_based, fit,
                                    plaintext[:50]))
                if fit > best_fit:
                    best_fit = fit
                    best_multiplier = multiplier
                    best_adder = adder
                    best_one_based = one_based
    logger.info('Affine break best fit with key {0}x+{1} ({2}) gives fit of '
                '{3} and decrypt starting: {4}'.format(
                    best_multiplier, best_adder, best_one_based, best_fit,
                    affine_decipher(sanitised_message, best_multiplier,
                                    best_adder, best_one_based)[:50]))
    return (best_multiplier, best_adder, best_one_based), best_fit

def keyword_break(message, wordlist=keywords, fitness=Pletters):
    """Breaks a keyword substitution cipher using a dictionary and
    frequency analysis.

    >>> keyword_break(keyword_encipher('this is a test message for the ' \
          'keyword decipherment', 'elephant', KeywordWrapAlphabet.from_last), \
          wordlist=['cat', 'elephant', 'kangaroo']) # doctest: +ELLIPSIS
    (('elephant', <KeywordWrapAlphabet.from_last: 2>), -52.834575011...)
    """
    best_keyword = ''
    best_wrap_alphabet = True
    best_fit = float("-inf")
    for wrap_alphabet in KeywordWrapAlphabet:
        for keyword in wordlist:
            plaintext = keyword_decipher(message, keyword, wrap_alphabet)
            fit = fitness(plaintext)
            logger.debug('Keyword break attempt using key {0} (wrap={1}) '
                         'gives fit of {2} and decrypt starting: {3}'.format(
                             keyword, wrap_alphabet, fit,
                             sanitise(plaintext)[:50]))
            if fit > best_fit:
                best_fit = fit
                best_keyword = keyword
                best_wrap_alphabet = wrap_alphabet
    logger.info('Keyword break best fit with key {0} (wrap={1}) gives fit of '
                '{2} and decrypt starting: {3}'.format(best_keyword,
                    best_wrap_alphabet, best_fit, sanitise(
                        keyword_decipher(message, best_keyword,
                                         best_wrap_alphabet))[:50]))
    return (best_keyword, best_wrap_alphabet), best_fit

def keyword_break_mp(message, wordlist=keywords, fitness=Pletters,
                     number_of_solutions=1, chunksize=500):
    """Breaks a keyword substitution cipher using a dictionary and
    frequency analysis

    >>> keyword_break_mp(keyword_encipher('this is a test message for the ' \
          'keyword decipherment', 'elephant', KeywordWrapAlphabet.from_last), \
          wordlist=['cat', 'elephant', 'kangaroo']) # doctest: +ELLIPSIS
    (('elephant', <KeywordWrapAlphabet.from_last: 2>), -52.834575011...)
    >>> keyword_break_mp(keyword_encipher('this is a test message for the ' \
          'keyword decipherment', 'elephant', KeywordWrapAlphabet.from_last), \
          wordlist=['cat', 'elephant', 'kangaroo'], \
          number_of_solutions=2) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    [(('elephant', <KeywordWrapAlphabet.from_last: 2>), -52.834575011...), 
    (('elephant', <KeywordWrapAlphabet.from_largest: 3>), -52.834575011...)]
    """
    with Pool() as pool:
        helper_args = [(message, word, wrap, fitness)
                       for word in wordlist
                       for wrap in KeywordWrapAlphabet]
        # Gotcha: the helper function here needs to be defined at the top level
        #   (limitation of Pool.starmap)
        breaks = pool.starmap(keyword_break_worker, helper_args, chunksize)
        if number_of_solutions == 1:
            return max(breaks, key=lambda k: k[1])
        else:
            return sorted(breaks, key=lambda k: k[1], reverse=True)[:number_of_solutions]

def keyword_break_worker(message, keyword, wrap_alphabet, fitness):
    plaintext = keyword_decipher(message, keyword, wrap_alphabet)
    fit = fitness(plaintext)
    logger.debug('Keyword break attempt using key {0} (wrap={1}) gives fit of '
                 '{2} and decrypt starting: {3}'.format(keyword, 
                     wrap_alphabet, fit, sanitise(plaintext)[:50]))
    return (keyword, wrap_alphabet), fit

def monoalphabetic_break_hillclimbing(message, max_iterations=10000000, 
        fitness=Pletters):
    ciphertext = unaccent(message).lower()
    alphabet = list(string.ascii_lowercase)
    random.shuffle(alphabet)
    alphabet = ''.join(alphabet)
    return monoalphabetic_break_hillclimbing_worker(ciphertext, alphabet,
                                                    max_iterations, fitness)

def monoalphabetic_break_hillclimbing_mp(message, workers=10, 
        max_iterations = 10000000, fitness=Pletters, chunksize=1):
    worker_args = []
    ciphertext = unaccent(message).lower()
    for i in range(workers):
        alphabet = list(string.ascii_lowercase)
        random.shuffle(alphabet)
        alphabet = ''.join(alphabet)
        worker_args.append((ciphertext, alphabet, max_iterations, fitness))
    with Pool() as pool:
        breaks = pool.starmap(monoalphabetic_break_hillclimbing_worker,
                              worker_args, chunksize)
    return max(breaks, key=lambda k: k[1])

def monoalphabetic_break_hillclimbing_worker(message, alphabet,
        max_iterations, fitness):
    def swap(letters, i, j):
        if i > j:
            i, j = j, i
        if i == j:
            return letters
        else:
            return (letters[:i] + letters[j] + letters[i+1:j] + letters[i] +
                    letters[j+1:])
    best_alphabet = alphabet
    best_fitness = float('-inf')
    for i in range(max_iterations):
        alphabet = swap(alphabet, random.randrange(26), random.randrange(26))
        cipher_translation = ''.maketrans(string.ascii_lowercase, alphabet)
        plaintext = message.translate(cipher_translation)
        if fitness(plaintext) > best_fitness:
            best_fitness = fitness(plaintext)
            best_alphabet = alphabet
            print(i, best_alphabet, best_fitness, plaintext)
    return best_alphabet, best_fitness


def vigenere_keyword_break_mp(message, wordlist=keywords, fitness=Pletters,
                              chunksize=500):
    """Breaks a vigenere cipher using a dictionary and frequency analysis.

    >>> vigenere_keyword_break_mp(vigenere_encipher(sanitise('this is a test ' \
             'message for the vigenere decipherment'), 'cat'), \
             wordlist=['cat', 'elephant', 'kangaroo']) # doctest: +ELLIPSIS
    ('cat', -52.947271216...)
    """
    with Pool() as pool:
        helper_args = [(message, word, fitness)
                       for word in wordlist]
        # Gotcha: the helper function here needs to be defined at the top level
        #   (limitation of Pool.starmap)
        breaks = pool.starmap(vigenere_keyword_break_worker, helper_args,
                              chunksize)
        return max(breaks, key=lambda k: k[1])
vigenere_keyword_break = vigenere_keyword_break_mp

def vigenere_keyword_break_worker(message, keyword, fitness):
    plaintext = vigenere_decipher(message, keyword)
    fit = fitness(plaintext)
    logger.debug('Vigenere keyword break attempt using key {0} gives fit of '
                 '{1} and decrypt starting: {2}'.format(keyword,
                     fit, sanitise(plaintext)[:50]))
    return keyword, fit


def vigenere_frequency_break(message, max_key_length=20, fitness=Pletters):
    """Breaks a Vigenere cipher with frequency analysis

    >>> vigenere_frequency_break(vigenere_encipher(sanitise("It is time to " \
            "run. She is ready and so am I. I stole Daniel's pocketbook this " \
            "afternoon when he left his jacket hanging on the easel in the " \
            "attic. I jump every time I hear a footstep on the stairs, " \
            "certain that the theft has been discovered and that I will " \
            "be caught. The SS officer visits less often now that he is " \
            "sure"), 'florence')) # doctest: +ELLIPSIS
    ('florence', -307.5473096791...)
    """
    def worker(message, key_length, fitness):
        splits = every_nth(sanitised_message, key_length)
        key = ''.join([chr(caesar_break(s)[0] + ord('a')) for s in splits])
        plaintext = vigenere_decipher(message, key)
        fit = fitness(plaintext)
        return key, fit
    sanitised_message = sanitise(message)
    results = starmap(worker, [(sanitised_message, i, fitness)
                               for i in range(1, max_key_length+1)])
    return max(results, key=lambda k: k[1])


def beaufort_frequency_break(message, max_key_length=20, fitness=Pletters):
    """Breaks a Beaufort cipher with frequency analysis

    >>> beaufort_frequency_break(beaufort_encipher(sanitise("It is time to " \
            "run. She is ready and so am I. I stole Daniel's pocketbook this " \
            "afternoon when he left his jacket hanging on the easel in the " \
            "attic. I jump every time I hear a footstep on the stairs, " \
            "certain that the theft has been discovered and that I will " \
            "be caught. The SS officer visits less often now " \
            "that he is sure"), 'florence')) # doctest: +ELLIPSIS
    ('florence', -307.5473096791...)
    """
    def worker(message, key_length, fitness):
        splits = every_nth(sanitised_message, key_length)
        key = ''.join([chr(-caesar_break(s)[0] % 26 + ord('a'))
                       for s in splits])
        plaintext = beaufort_decipher(message, key)
        fit = fitness(plaintext)
        return key, fit
    sanitised_message = sanitise(message)
    results = starmap(worker, [(sanitised_message, i, fitness)
                               for i in range(1, max_key_length+1)])
    return max(results, key=lambda k: k[1])


def column_transposition_break_mp(message, translist=transpositions,
                                  fitness=Pbigrams, chunksize=500):
    """Breaks a column transposition cipher using a dictionary and
    n-gram frequency analysis

    >>> column_transposition_break_mp(column_transposition_encipher(sanitise( \
            "It is a truth universally acknowledged, that a single man in \
             possession of a good fortune, must be in want of a wife. However \
             little known the feelings or views of such a man may be on his \
             first entering a neighbourhood, this truth is so well fixed in \
             the minds of the surrounding families, that he is considered the \
             rightful property of some one or other of their daughters."), \
        'encipher'), \
        translist={(2, 0, 5, 3, 1, 4, 6): ['encipher'], \
                   (5, 0, 6, 1, 3, 4, 2): ['fourteen'], \
                   (6, 1, 0, 4, 5, 3, 2): ['keyword']}) # doctest: +ELLIPSIS
    (((2, 0, 5, 3, 1, 4, 6), False, False), -709.4646722...)
    >>> column_transposition_break_mp(column_transposition_encipher(sanitise( \
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
        fitness=Ptrigrams) # doctest: +ELLIPSIS
    (((2, 0, 5, 3, 1, 4, 6), False, False), -997.0129085...)
    """
    with Pool() as pool:
        helper_args = [(message, trans, fillcolumnwise, emptycolumnwise,
                        fitness)
                       for trans in translist.keys()
                       for fillcolumnwise in [True, False]
                       for emptycolumnwise in [True, False]]
        # Gotcha: the helper function here needs to be defined at the top level
        #   (limitation of Pool.starmap)
        breaks = pool.starmap(column_transposition_break_worker,
                              helper_args, chunksize) 
        return max(breaks, key=lambda k: k[1])
column_transposition_break = column_transposition_break_mp

def column_transposition_break_worker(message, transposition,
        fillcolumnwise, emptycolumnwise, fitness):
    plaintext = column_transposition_decipher(message, transposition,
        fillcolumnwise=fillcolumnwise, emptycolumnwise=emptycolumnwise)
    fit = fitness(sanitise(plaintext))
    logger.debug('Column transposition break attempt using key {0} '
                         'gives fit of {1} and decrypt starting: {2}'.format(
                             transposition, fit, 
                             sanitise(plaintext)[:50]))
    return (transposition, fillcolumnwise, emptycolumnwise), fit


def scytale_break_mp(message, max_key_length=20,
                     fitness=Pbigrams, chunksize=500):
    """Breaks a scytale cipher using a range of lengths and
    n-gram frequency analysis

    >>> scytale_break_mp(scytale_encipher(sanitise( \
            "It is a truth universally acknowledged, that a single man in \
             possession of a good fortune, must be in want of a wife. However \
             little known the feelings or views of such a man may be on his \
             first entering a neighbourhood, this truth is so well fixed in \
             the minds of the surrounding families, that he is considered the \
             rightful property of some one or other of their daughters."), \
        5)) # doctest: +ELLIPSIS
    (5, -709.4646722...)
    >>> scytale_break_mp(scytale_encipher(sanitise( \
            "It is a truth universally acknowledged, that a single man in \
             possession of a good fortune, must be in want of a wife. However \
             little known the feelings or views of such a man may be on his \
             first entering a neighbourhood, this truth is so well fixed in \
             the minds of the surrounding families, that he is considered the \
             rightful property of some one or other of their daughters."), \
        5), \
        fitness=Ptrigrams) # doctest: +ELLIPSIS
    (5, -997.0129085...)
    """
    with Pool() as pool:
        helper_args = [(message, trans, False, True, fitness)
            for trans in
                [[col for col in range(math.ceil(len(message)/rows))]
                    for rows in range(1,max_key_length+1)]]
        # Gotcha: the helper function here needs to be defined at the top level
        #   (limitation of Pool.starmap)
        breaks = pool.starmap(column_transposition_break_worker,
                              helper_args, chunksize)
        best = max(breaks, key=lambda k: k[1])
        return math.trunc(len(message) / len(best[0][0])), best[1]
scytale_break = scytale_break_mp


def railfence_break(message, max_key_length=20,
                     fitness=Pbigrams, chunksize=500):
    """Breaks a railfence cipher using a range of lengths and
    n-gram frequency analysis

    >>> railfence_break(railfence_encipher(sanitise( \
            "It is a truth universally acknowledged, that a single man in \
             possession of a good fortune, must be in want of a wife. However \
             little known the feelings or views of such a man may be on his \
             first entering a neighbourhood, this truth is so well fixed in \
             the minds of the surrounding families, that he is considered the \
             rightful property of some one or other of their daughters."), \
        7)) # doctest: +ELLIPSIS
    (7, -709.46467226...)
    >>> railfence_break(railfence_encipher(sanitise( \
            "It is a truth universally acknowledged, that a single man in \
             possession of a good fortune, must be in want of a wife. However \
             little known the feelings or views of such a man may be on his \
             first entering a neighbourhood, this truth is so well fixed in \
             the minds of the surrounding families, that he is considered the \
             rightful property of some one or other of their daughters."), \
        7), \
        fitness=Ptrigrams) # doctest: +ELLIPSIS
    (7, -997.0129085...)
    """
    def worker(message, height, fitness):
        plaintext = railfence_decipher(message, height)
        fit = fitness(plaintext)
        return height, fit
        
    sanitised_message = sanitise(message)
    results = starmap(worker, [(sanitised_message, i, fitness)
                               for i in range(2, max_key_length+1)])
    return max(results, key=lambda k: k[1])


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
    x = range(len(freqs.keys()))
    y = [freqs[l] for l in sorted(freqs.keys(), key=sort_key)]
    f = plt.figure()
    ax = f.add_axes([0.1, 0.1, 0.9, 0.9])
    ax.bar(x, y, align='center')
    ax.set_xticks(x)
    ax.set_xticklabels(sorted(freqs.keys(), key=sort_key))
    f.show()


if __name__ == "__main__":
    import doctest
    doctest.testmod()

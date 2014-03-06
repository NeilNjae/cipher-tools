import string
import collections
import norms
import logging
from itertools import zip_longest, cycle, permutations
from segment import segment
from multiprocessing import Pool
from math import log10

import matplotlib.pyplot as plt

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
    >>> sorted(frequencies(sanitise('The Quick BROWN fox jumped! over... ' \
         'the (9lazy) DOG')).items()) # doctest: +NORMALIZE_WHITESPACE
    [('a', 1), ('b', 1), ('c', 1), ('d', 2), ('e', 4), ('f', 1), ('g', 1), 
     ('h', 2), ('i', 1), ('j', 1), ('k', 1), ('l', 1), ('m', 1), ('n', 1), 
     ('o', 4), ('p', 1), ('q', 1), ('r', 2), ('t', 2), ('u', 2), ('v', 1), 
     ('w', 1), ('x', 1), ('y', 1), ('z', 1)]
    >>> frequencies('abcdefabcdef')['x']
    0
    """
    #counts = collections.defaultdict(int)
    #for c in text: 
    #    counts[c] += 1
    #return counts
    return collections.Counter(c for c in text)


def caesar_break(message, fitness=Pletters):
    """Breaks a Caesar cipher using frequency analysis
    
    >>> caesar_break('ibxcsyorsaqcheyklxivoexlevmrimwxsfiqevvmihrsasrxliwyrh' \
          'ecjsppsamrkwleppfmergefifvmhixscsymjcsyqeoixlm') # doctest: +ELLIPSIS
    (4, -130.849890899...)
    >>> caesar_break('wxwmaxdgheetgwuxztgptedbgznitgwwhpguxyhkxbmhvvtlbhgtee' \
          'raxlmhiixweblmxgxwmhmaxybkbgztgwztsxwbgmxgmert') # doctest: +ELLIPSIS
    (19, -128.82516920...)
    >>> caesar_break('yltbbqnqnzvguvaxurorgenafsbezqvagbnornfgsbevpnaabjurer' \
          'svaquvzyvxrnznazlybequrvfohgriraabjtbaruraprur') # doctest: +ELLIPSIS
    (13, -126.25233502...)
    """
    sanitised_message = sanitise(message)
    best_shift = 0
    best_fit = float('-inf')
    for shift in range(26):
        plaintext = caesar_decipher(sanitised_message, shift)
        fit = fitness(plaintext)
        logger.debug('Caesar break attempt using key {0} gives fit of {1} '
                      'and decrypt starting: {2}'.format(shift, fit, plaintext[:50]))
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
    ((15, 22, True), -340.611412245...)
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
    logger.info('Affine break best fit with key {0}x+{1} ({2}) gives fit of {3} '
                'and decrypt starting: {4}'.format(
                    best_multiplier, best_adder, best_one_based, best_fit, 
                    affine_decipher(sanitised_message, best_multiplier, 
                        best_adder, best_one_based)[:50]))
    return (best_multiplier, best_adder, best_one_based), best_fit

def keyword_break(message, wordlist=keywords, fitness=Pletters):
    """Breaks a keyword substitution cipher using a dictionary and 
    frequency analysis

    >>> keyword_break(keyword_encipher('this is a test message for the ' \
          'keyword decipherment', 'elephant', 1), \
          wordlist=['cat', 'elephant', 'kangaroo']) # doctest: +ELLIPSIS
    (('elephant', 1), -52.8345642265...)
    """
    best_keyword = ''
    best_wrap_alphabet = True
    best_fit = float("-inf")
    for wrap_alphabet in range(3):
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

def keyword_break_mp(message, wordlist=keywords, fitness=Pletters, chunksize=500):
    """Breaks a keyword substitution cipher using a dictionary and 
    frequency analysis

    >>> keyword_break_mp(keyword_encipher('this is a test message for the ' \
          'keyword decipherment', 'elephant', 1), \
          wordlist=['cat', 'elephant', 'kangaroo']) # doctest: +ELLIPSIS
    (('elephant', 1), -52.834564226507...)
    """
    with Pool() as pool:
        helper_args = [(message, word, wrap, fitness) 
                       for word in wordlist for wrap in range(3)]
        # Gotcha: the helper function here needs to be defined at the top level 
        #   (limitation of Pool.starmap)
        breaks = pool.starmap(keyword_break_worker, helper_args, chunksize) 
        return max(breaks, key=lambda k: k[1])

def keyword_break_worker(message, keyword, wrap_alphabet, fitness):
    plaintext = keyword_decipher(message, keyword, wrap_alphabet)
    fit = fitness(plaintext)
    logger.debug('Keyword break attempt using key {0} (wrap={1}) gives fit of '
                 '{2} and decrypt starting: {3}'.format(keyword, 
                     wrap_alphabet, fit, sanitise(plaintext)[:50]))
    return (keyword, wrap_alphabet), fit

def scytale_break(message, fitness=Pbigrams):
    """Breaks a Scytale cipher
    
    >>> scytale_break('tfeulchtrtteehwahsdehneoifeayfsondmwpltmaoalhikotoere' \
           'dcweatehiplwxsnhooacgorrcrcraotohsgullasenylrendaianeplscdriioto' \
           'aek') # doctest: +ELLIPSIS
    (6, -281.276219108...)
    """
    best_key = 0
    best_fit = float("-inf")
    for key in range(1, 20):
        if len(message) % key == 0:
            plaintext = scytale_decipher(message, key)
            fit = fitness(sanitise(plaintext))
            logger.debug('Scytale break attempt using key {0} gives fit of '
                         '{1} and decrypt starting: {2}'.format(key, 
                             fit, sanitise(plaintext)[:50]))
            if fit > best_fit:
                best_fit = fit
                best_key = key
    logger.info('Scytale break best fit with key {0} gives fit of {1} and '
                'decrypt starting: {2}'.format(best_key, best_fit, 
                    sanitise(scytale_decipher(message, best_key))[:50]))
    return best_key, best_fit


def column_transposition_break_mp(message, translist=transpositions, 
                     fitness=Pbigrams, chunksize=500):
    """Breaks a column transposition cipher using a dictionary and 
    n-gram frequency analysis
    """
    # >>> column_transposition_break_mp(column_transposition_encipher(sanitise( \
    #         "It is a truth universally acknowledged, that a single man in \
    #          possession of a good fortune, must be in want of a wife. However \
    #          little known the feelings or views of such a man may be on his \
    #          first entering a neighbourhood, this truth is so well fixed in the \
    #          minds of the surrounding families, that he is considered the \
    #          rightful property of some one or other of their daughters."), \
    #     'encipher'), \
    #     translist={(2, 0, 5, 3, 1, 4, 6): ['encipher'], \
    #                (5, 0, 6, 1, 3, 4, 2): ['fourteen'], \
    #                (6, 1, 0, 4, 5, 3, 2): ['keyword']}) # doctest: +ELLIPSIS
    # (((2, 0, 5, 3, 1, 4, 6), False), 0.0628106372...)
    # >>> column_transposition_break_mp(column_transposition_encipher(sanitise( \
    #         "It is a truth universally acknowledged, that a single man in \
    #          possession of a good fortune, must be in want of a wife. However \
    #          little known the feelings or views of such a man may be on his \
    #          first entering a neighbourhood, this truth is so well fixed in the \
    #          minds of the surrounding families, that he is considered the \
    #          rightful property of some one or other of their daughters."), \
    #     'encipher'), \
    #     translist={(2, 0, 5, 3, 1, 4, 6): ['encipher'], \
    #                (5, 0, 6, 1, 3, 4, 2): ['fourteen'], \
    #                (6, 1, 0, 4, 5, 3, 2): ['keyword']}, \
    #     target_counts=normalised_english_trigram_counts) # doctest: +ELLIPSIS
    # (((2, 0, 5, 3, 1, 4, 6), False), 0.0592259560...)
    # """
    with Pool() as pool:
        helper_args = [(message, trans, columnwise, fitness) 
                       for trans in translist.keys() 
                       for columnwise in [True, False]]
        # Gotcha: the helper function here needs to be defined at the top level 
        #   (limitation of Pool.starmap)
        breaks = pool.starmap(column_transposition_break_worker, 
          helper_args, chunksize) 
        return max(breaks, key=lambda k: k[1])
column_transposition_break = column_transposition_break_mp

def column_transposition_break_worker(message, transposition, columnwise, 
                                        fitness):
    plaintext = column_transposition_decipher(message, transposition, columnwise=columnwise)
    fit = fitness(sanitise(plaintext))
    logger.debug('Column transposition break attempt using key {0} '
                         'gives fit of {1} and decrypt starting: {2}'.format(
                             transposition, fit, 
                             sanitise(plaintext)[:50]))
    return (transposition, columnwise), fit


def transposition_break_exhaustive(message, fitness=Pbigrams):
    best_transposition = ''
    best_pw = float('-inf')
    for keylength in range(1, 21):
        if len(message) % keylength == 0:
            for transposition in permutations(range(keylength)):
                for columnwise in [True, False]:
                    plaintext = column_transposition_decipher(message, 
                        transposition, columnwise=columnwise)
                    fit=fitness(plaintext)
                    logger.debug('Column transposition break attempt using key {0} {1} '
                         'gives fit of {2} and decrypt starting: {3}'.format(
                             transposition, columnwise, pw, 
                             sanitise(plaintext)[:50]))
                    if fit > best_fit:
                        best_transposition = transposition
                        best_columnwise = columnwise
                        best_fit = fit
    return (best_transposition, best_columnwise), best_pw


def vigenere_keyword_break(message, wordlist=keywords, fitness=Pletters):
    """Breaks a vigenere cipher using a dictionary and 
    frequency analysis
    
    >>> vigenere_keyword_break(vigenere_encipher(sanitise('this is a test ' \
             'message for the vigenere decipherment'), 'cat'), \
             wordlist=['cat', 'elephant', 'kangaroo']) # doctest: +ELLIPSIS
    ('cat', -52.9479167030...)
    """
    best_keyword = ''
    best_fit = float("-inf")
    for keyword in wordlist:
        plaintext = vigenere_decipher(message, keyword)
        fit = fitness(plaintext)
        logger.debug('Vigenere break attempt using key {0} '
                         'gives fit of {1} and decrypt starting: {2}'.format(
                             keyword, fit, 
                             sanitise(plaintext)[:50]))
        if fit > best_fit:
            best_fit = fit
            best_keyword = keyword
    logger.info('Vigenere break best fit with key {0} gives fit '
                'of {1} and decrypt starting: {2}'.format(best_keyword, 
                    best_fit, sanitise(
                        vigenere_decipher(message, best_keyword))[:50]))
    return best_keyword, best_fit

def vigenere_keyword_break_mp(message, wordlist=keywords, fitness=Pletters, 
                     chunksize=500):
    """Breaks a vigenere cipher using a dictionary and 
    frequency analysis

    >>> vigenere_keyword_break_mp(vigenere_encipher(sanitise('this is a test ' \
             'message for the vigenere decipherment'), 'cat'), \
             wordlist=['cat', 'elephant', 'kangaroo']) # doctest: +ELLIPSIS
    ('cat', -52.9479167030...)
    """
    with Pool() as pool:
        helper_args = [(message, word, fitness) 
                       for word in wordlist]
        # Gotcha: the helper function here needs to be defined at the top level 
        #   (limitation of Pool.starmap)
        breaks = pool.starmap(vigenere_keyword_break_worker, helper_args, chunksize) 
        return max(breaks, key=lambda k: k[1])

def vigenere_keyword_break_worker(message, keyword, fitness):
    plaintext = vigenere_decipher(message, keyword)
    fit = fitness(plaintext)
    logger.debug('Vigenere keyword break attempt using key {0} gives fit of '
                 '{1} and decrypt starting: {2}'.format(keyword, 
                     fit, sanitise(plaintext)[:50]))
    return keyword, fit



def vigenere_frequency_break(message, fitness=Pletters):
    """Breaks a Vigenere cipher with frequency analysis

    >>> vigenere_frequency_break(vigenere_encipher(sanitise("It is time to " \
            "run. She is ready and so am I. I stole Daniel's pocketbook this " \
            "afternoon when he left his jacket hanging on the easel in the " \
            "attic. I jump every time I hear a footstep on the stairs, " \
            "certain that the theft has been discovered and that I will " \
            "be caught. The SS officer visits less often now that he is " \
            "sure"), 'florence')) # doctest: +ELLIPSIS
    ('florence', -307.5549865898...)
    """
    best_fit = float("-inf")
    best_key = ''
    sanitised_message = sanitise(message)
    for trial_length in range(1, 20):
        splits = every_nth(sanitised_message, trial_length)
        key = ''.join([chr(caesar_break(s)[0] + ord('a')) for s in splits])
        plaintext = vigenere_decipher(sanitised_message, key)
        fit = fitness(plaintext)
        logger.debug('Vigenere key length of {0} ({1}) gives fit of {2}'.
                     format(trial_length, key, fit))
        if fit > best_fit:
            best_fit = fit
            best_key = key
    logger.info('Vigenere break best fit with key {0} gives fit '
                'of {1} and decrypt starting: {2}'.format(best_key, 
                    best_fit, sanitise(
                        vigenere_decipher(message, best_key))[:50]))
    return best_key, best_fit

def beaufort_frequency_break(message, fitness=Pletters):
    """Breaks a Beaufort cipher with frequency analysis

    >>> beaufort_frequency_break(beaufort_encipher(sanitise("It is time to " \
            "run. She is ready and so am I. I stole Daniel's pocketbook this " \
            "afternoon when he left his jacket hanging on the easel in the " \
            "attic. I jump every time I hear a footstep on the stairs, " \
            "certain that the theft has been discovered and that I will " \
            "be caught. The SS officer visits less often now " \
            "that he is sure"), 'florence')) # doctest: +ELLIPSIS
    ('florence', -307.5549865898...)
    """
    best_fit = float("-inf")
    best_key = ''
    sanitised_message = sanitise(message)
    for trial_length in range(1, 20):
        splits = every_nth(sanitised_message, trial_length)
        key = ''.join([chr(-caesar_break(s)[0] % 26 + ord('a')) for s in splits])
        plaintext = beaufort_decipher(sanitised_message, key)
        fit = fitness(plaintext)
        logger.debug('Beaufort key length of {0} ({1}) gives fit of {2}'.
                     format(trial_length, key, fit))
        if fit > best_fit:
            best_fit = fit
            best_key = key
    logger.info('Beaufort break best fit with key {0} gives fit '
                'of {1} and decrypt starting: {2}'.format(best_key, 
                    best_fit, sanitise(
                        beaufort_decipher(message, best_key))[:50]))
    return best_key, best_fit



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


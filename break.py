import string
import collections
import norms
import logging
from itertools import zip_longest, cycle
from segment import segment
from multiprocessing import Pool

from cipher import *

# To time a run:
#
# import timeit
# c5a = open('2012/5a.ciphertext', 'r').read()
# timeit.timeit('keyword_break(c5a)', setup='gc.enable() ; from __main__ import c5a ; from cipher import keyword_break', number=1)
# timeit.repeat('keyword_break_mp(c5a, chunksize=500)', setup='gc.enable() ; from __main__ import c5a ; from cipher import keyword_break_mp', repeat=5, number=1)


english_counts = collections.defaultdict(int)
with open('count_1l.txt', 'r') as f:
    for line in f:
        (letter, count) = line.split("\t")
        english_counts[letter] = int(count)
normalised_english_counts = norms.normalise(english_counts)

english_bigram_counts = collections.defaultdict(int)
with open('count_2l.txt', 'r') as f:
    for line in f:
        (bigram, count) = line.split("\t")
        english_bigram_counts[bigram] = int(count)
normalised_english_bigram_counts = norms.normalise(english_bigram_counts)

english_trigram_counts = collections.defaultdict(int)
with open('count_3l.txt', 'r') as f:
    for line in f:
        (trigram, count) = line.split("\t")
        english_trigram_counts[trigram] = int(count)
normalised_english_trigram_counts = norms.normalise(english_trigram_counts)


with open('words.txt', 'r') as f:
    keywords = [line.rstrip() for line in f]

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
letter_frequencies = frequencies



def caesar_break(message, 
                 metric=norms.euclidean_distance, 
                 target_counts=normalised_english_counts, 
                 message_frequency_scaling=norms.normalise):
    """Breaks a Caesar cipher using frequency analysis
    
    >>> caesar_break('ibxcsyorsaqcheyklxivoexlevmrimwxsfiqevvmihrsasrxliwyrh' \
          'ecjsppsamrkwleppfmergefifvmhixscsymjcsyqeoixlm') # doctest: +ELLIPSIS
    (4, 0.31863952890183...)
    >>> caesar_break('wxwmaxdgheetgwuxztgptedbgznitgwwhpguxyhkxbmhvvtlbhgtee' \
          'raxlmhiixweblmxgxwmhmaxybkbgztgwztsxwbgmxgmert') # doctest: +ELLIPSIS
    (19, 0.42152901235832...)
    >>> caesar_break('yltbbqnqnzvguvaxurorgenafsbezqvagbnornfgsbevpnaabjurer' \
          'svaquvzyvxrnznazlybequrvfohgriraabjtbaruraprur') # doctest: +ELLIPSIS
    (13, 0.316029208075451...)
    """
    sanitised_message = sanitise(message)
    best_shift = 0
    best_fit = float("inf")
    for shift in range(26):
        plaintext = caesar_decipher(sanitised_message, shift)
        counts = message_frequency_scaling(letter_frequencies(plaintext))
        fit = metric(target_counts, counts)
        logger.debug('Caesar break attempt using key {0} gives fit of {1} '
                      'and decrypt starting: {2}'.format(shift, fit, plaintext[:50]))
        if fit < best_fit:
            best_fit = fit
            best_shift = shift
    logger.info('Caesar break best fit: key {0} gives fit of {1} and '
                'decrypt starting: {2}'.format(best_shift, best_fit, 
                    caesar_decipher(sanitised_message, best_shift)[:50]))
    return best_shift, best_fit

def affine_break(message, 
                 metric=norms.euclidean_distance, 
                 target_counts=normalised_english_counts, 
                 message_frequency_scaling=norms.normalise):
    """Breaks an affine cipher using frequency analysis
    
    >>> affine_break('lmyfu bkuusd dyfaxw claol psfaom jfasd snsfg jfaoe ls ' \
          'omytd jlaxe mh jm bfmibj umis hfsul axubafkjamx. ls kffkxwsd jls ' \
          'ofgbjmwfkiu olfmxmtmwaokttg jlsx ls kffkxwsd jlsi zg tsxwjl. jlsx ' \
          'ls umfjsd jlsi zg hfsqysxog. ls dmmdtsd mx jls bats mh bkbsf. ls ' \
          'bfmctsd kfmyxd jls lyj, mztanamyu xmc jm clm cku tmmeaxw kj lai kxd ' \
          'clm ckuxj.') # doctest: +ELLIPSIS
    ((15, 22, True), 0.23570361818655...)
    """
    sanitised_message = sanitise(message)
    best_multiplier = 0
    best_adder = 0
    best_one_based = True
    best_fit = float("inf")
    for one_based in [True, False]:
        for multiplier in range(1, 26, 2):
            for adder in range(26):
                plaintext = affine_decipher(sanitised_message, 
                                            multiplier, adder, one_based)
                counts = message_frequency_scaling(letter_frequencies(plaintext))
                fit = metric(target_counts, counts)
                logger.debug('Affine break attempt using key {0}x+{1} ({2}) '
                             'gives fit of {3} and decrypt starting: {4}'.
                             format(multiplier, adder, one_based, fit, 
                                    plaintext[:50]))
                if fit < best_fit:
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

def keyword_break(message, 
                  wordlist=keywords, 
                  metric=norms.euclidean_distance, 
                  target_counts=normalised_english_counts, 
                  message_frequency_scaling=norms.normalise):
    """Breaks a keyword substitution cipher using a dictionary and 
    frequency analysis

    >>> keyword_break(keyword_encipher('this is a test message for the ' \
          'keyword decipherment', 'elephant', 1), \
          wordlist=['cat', 'elephant', 'kangaroo']) # doctest: +ELLIPSIS
    (('elephant', 1), 0.41643991598441...)
    """
    best_keyword = ''
    best_wrap_alphabet = True
    best_fit = float("inf")
    for wrap_alphabet in range(3):
        for keyword in wordlist:
            plaintext = keyword_decipher(message, keyword, wrap_alphabet)
            counts = message_frequency_scaling(letter_frequencies(plaintext))
            fit = metric(target_counts, counts)
            logger.debug('Keyword break attempt using key {0} (wrap={1}) '
                         'gives fit of {2} and decrypt starting: {3}'.format(
                             keyword, wrap_alphabet, fit, 
                             sanitise(plaintext)[:50]))
            if fit < best_fit:
                best_fit = fit
                best_keyword = keyword
                best_wrap_alphabet = wrap_alphabet
    logger.info('Keyword break best fit with key {0} (wrap={1}) gives fit of '
                '{2} and decrypt starting: {3}'.format(best_keyword, 
                    best_wrap_alphabet, best_fit, sanitise(
                        keyword_decipher(message, best_keyword, 
                                         best_wrap_alphabet))[:50]))
    return (best_keyword, best_wrap_alphabet), best_fit

def keyword_break_mp(message, 
                     wordlist=keywords, 
                     metric=norms.euclidean_distance, 
                     target_counts=normalised_english_counts, 
                     message_frequency_scaling=norms.normalise, 
                     chunksize=500):
    """Breaks a keyword substitution cipher using a dictionary and 
    frequency analysis

    >>> keyword_break_mp(keyword_encipher('this is a test message for the ' \
          'keyword decipherment', 'elephant', 1), \
          wordlist=['cat', 'elephant', 'kangaroo']) # doctest: +ELLIPSIS
    (('elephant', 1), 0.41643991598441...)
    """
    with Pool() as pool:
        helper_args = [(message, word, wrap, metric, target_counts, 
                        message_frequency_scaling) 
                       for word in wordlist for wrap in range(3)]
        # Gotcha: the helper function here needs to be defined at the top level 
        #   (limitation of Pool.starmap)
        breaks = pool.starmap(keyword_break_worker, helper_args, chunksize) 
        return min(breaks, key=lambda k: k[1])

def keyword_break_worker(message, keyword, wrap_alphabet, metric, target_counts, 
                      message_frequency_scaling):
    plaintext = keyword_decipher(message, keyword, wrap_alphabet)
    counts = message_frequency_scaling(letter_frequencies(plaintext))
    fit = metric(target_counts, counts)
    logger.debug('Keyword break attempt using key {0} (wrap={1}) gives fit of '
                 '{2} and decrypt starting: {3}'.format(keyword, 
                     wrap_alphabet, fit, sanitise(plaintext)[:50]))
    return (keyword, wrap_alphabet), fit

def scytale_break(message, 
                  metric=norms.euclidean_distance, 
                  target_counts=normalised_english_bigram_counts, 
                  message_frequency_scaling=norms.normalise):
    """Breaks a Scytale cipher
    
    >>> scytale_break('tfeulchtrtteehwahsdehneoifeayfsondmwpltmaoalhikotoere' \
           'dcweatehiplwxsnhooacgorrcrcraotohsgullasenylrendaianeplscdriioto' \
           'aek') # doctest: +ELLIPSIS
    (6, 0.83453041115025...)
    """
    best_key = 0
    best_fit = float("inf")
    ngram_length = len(next(iter(target_counts.keys())))
    for key in range(1, 20):
        if len(message) % key == 0:
            plaintext = scytale_decipher(message, key)
            counts = message_frequency_scaling(frequencies(
                         ngrams(sanitise(plaintext), ngram_length)))
            fit = metric(target_counts, counts)
            logger.debug('Scytale break attempt using key {0} gives fit of '
                         '{1} and decrypt starting: {2}'.format(key, 
                             fit, sanitise(plaintext)[:50]))
            if fit < best_fit:
                best_fit = fit
                best_key = key
    logger.info('Scytale break best fit with key {0} gives fit of {1} and '
                'decrypt starting: {2}'.format(best_key, best_fit, 
                    sanitise(scytale_decipher(message, best_key))[:50]))
    return best_key, best_fit

def column_transposition_break(message, 
                  translist=transpositions, 
                  metric=norms.euclidean_distance, 
                  target_counts=normalised_english_bigram_counts, 
                  message_frequency_scaling=norms.normalise):
    """Breaks a column transposition cipher using a dictionary and 
    n-gram frequency analysis

    >>> column_transposition_break(column_transposition_encipher(sanitise( \
        "Turing's homosexuality resulted in a criminal prosecution in 1952, \
        when homosexual acts were still illegal in the United Kingdom. "), \
        'encipher'), \
        translist={(2, 0, 5, 3, 1, 4, 6): ['encipher'], \
                   (5, 0, 6, 1, 3, 4, 2): ['fourteen'], \
                   (6, 1, 0, 4, 5, 3, 2): ['keyword']}) # doctest: +ELLIPSIS
    ((2, 0, 5, 3, 1, 4, 6), 0.898128626285...)
    >>> column_transposition_break(column_transposition_encipher(sanitise( \
        "Turing's homosexuality resulted in a criminal prosecution in 1952, " \
        "when homosexual acts were still illegal in the United Kingdom."), \
        'encipher'), \
        translist={(2, 0, 5, 3, 1, 4, 6): ['encipher'], \
                   (5, 0, 6, 1, 3, 4, 2): ['fourteen'], \
                   (6, 1, 0, 4, 5, 3, 2): ['keyword']}, \
        target_counts=normalised_english_trigram_counts) # doctest: +ELLIPSIS
    ((2, 0, 5, 3, 1, 4, 6), 1.1958792913127...)
    """
    best_transposition = ''
    best_fit = float("inf")
    ngram_length = len(next(iter(target_counts.keys())))
    for transposition in translist.keys():
        if len(message) % len(transposition) == 0:
            plaintext = column_transposition_decipher(message, transposition)
            counts = message_frequency_scaling(frequencies(
                         ngrams(sanitise(plaintext), ngram_length)))
            fit = metric(target_counts, counts)
            logger.debug('Column transposition break attempt using key {0} '
                         'gives fit of {1} and decrypt starting: {2}'.format(
                             translist[transposition][0], fit, 
                             sanitise(plaintext)[:50]))
            if fit < best_fit:
                best_fit = fit
                best_transposition = transposition
    logger.info('Column transposition break best fit with key {0} gives fit '
                'of {1} and decrypt starting: {2}'.format(
                    translist[best_transposition][0], 
                    best_fit, sanitise(
                        column_transposition_decipher(message, 
                            best_transposition))[:50]))
    return best_transposition, best_fit


def column_transposition_break_mp(message, 
                     translist=transpositions, 
                     metric=norms.euclidean_distance, 
                     target_counts=normalised_english_bigram_counts, 
                     message_frequency_scaling=norms.normalise, 
                     chunksize=500):
    """Breaks a column transposition cipher using a dictionary and 
    n-gram frequency analysis

    >>> column_transposition_break_mp(column_transposition_encipher(sanitise( \
        "Turing's homosexuality resulted in a criminal prosecution in 1952, \
        when homosexual acts were still illegal in the United Kingdom. "), \
        'encipher'), \
        translist={(2, 0, 5, 3, 1, 4, 6): ['encipher'], \
                   (5, 0, 6, 1, 3, 4, 2): ['fourteen'], \
                   (6, 1, 0, 4, 5, 3, 2): ['keyword']}) # doctest: +ELLIPSIS
    ((2, 0, 5, 3, 1, 4, 6), 0.898128626285...)
    >>> column_transposition_break_mp(column_transposition_encipher(sanitise( \
        "Turing's homosexuality resulted in a criminal prosecution in 1952, " \
        "when homosexual acts were still illegal in the United Kingdom."), \
        'encipher'), \
        translist={(2, 0, 5, 3, 1, 4, 6): ['encipher'], \
                   (5, 0, 6, 1, 3, 4, 2): ['fourteen'], \
                   (6, 1, 0, 4, 5, 3, 2): ['keyword']}, \
        target_counts=normalised_english_trigram_counts) # doctest: +ELLIPSIS
    ((2, 0, 5, 3, 1, 4, 6), 1.1958792913127...)
    """
    ngram_length = len(next(iter(target_counts.keys())))
    with Pool() as pool:
        helper_args = [(message, trans, metric, target_counts, ngram_length,
                        message_frequency_scaling) 
                       for trans in translist.keys()]
        # Gotcha: the helper function here needs to be defined at the top level 
        #   (limitation of Pool.starmap)
        breaks = pool.starmap(column_transposition_break_worker, helper_args, chunksize) 
        return min(breaks, key=lambda k: k[1])

def column_transposition_break_worker(message, transposition, metric, target_counts, 
                      ngram_length, message_frequency_scaling):
    plaintext = column_transposition_decipher(message, transposition)
    counts = message_frequency_scaling(frequencies(
                         ngrams(sanitise(plaintext), ngram_length)))
    fit = metric(target_counts, counts)
    logger.debug('Column transposition break attempt using key {0} '
                         'gives fit of {1} and decrypt starting: {2}'.format(
                             transposition, fit, 
                             sanitise(plaintext)[:50]))
    return transposition, fit

def vigenere_keyword_break(message, 
                  wordlist=keywords, 
                  metric=norms.euclidean_distance, 
                  target_counts=normalised_english_counts, 
                  message_frequency_scaling=norms.normalise):
    """Breaks a vigenere cipher using a dictionary and 
    frequency analysis
    
    >>> vigenere_keyword_break(vigenere_encipher(sanitise('this is a test ' \
             'message for the vigenere decipherment'), 'cat'), \
             wordlist=['cat', 'elephant', 'kangaroo']) # doctest: +ELLIPSIS
    ('cat', 0.4950195952826...)
    """
    best_keyword = ''
    best_fit = float("inf")
    for keyword in wordlist:
        plaintext = vigenere_decipher(message, keyword)
        counts = message_frequency_scaling(letter_frequencies(plaintext))
        fit = metric(target_counts, counts)
        logger.debug('Vigenere break attempt using key {0} '
                         'gives fit of {1} and decrypt starting: {2}'.format(
                             keyword, fit, 
                             sanitise(plaintext)[:50]))
        if fit < best_fit:
            best_fit = fit
            best_keyword = keyword
    logger.info('Vigenere break best fit with key {0} gives fit '
                'of {1} and decrypt starting: {2}'.format(best_keyword, 
                    best_fit, sanitise(
                        vigenere_decipher(message, best_keyword))[:50]))
    return best_keyword, best_fit

def vigenere_keyword_break_mp(message, 
                     wordlist=keywords, 
                     metric=norms.euclidean_distance, 
                     target_counts=normalised_english_counts, 
                     message_frequency_scaling=norms.normalise, 
                     chunksize=500):
    """Breaks a vigenere cipher using a dictionary and 
    frequency analysis

    >>> vigenere_keyword_break_mp(vigenere_encipher(sanitise('this is a test ' \
             'message for the vigenere decipherment'), 'cat'), \
             wordlist=['cat', 'elephant', 'kangaroo']) # doctest: +ELLIPSIS
    ('cat', 0.4950195952826...)
    """
    with Pool() as pool:
        helper_args = [(message, word, metric, target_counts, 
                        message_frequency_scaling) 
                       for word in wordlist]
        # Gotcha: the helper function here needs to be defined at the top level 
        #   (limitation of Pool.starmap)
        breaks = pool.starmap(vigenere_keyword_break_worker, helper_args, chunksize) 
        return min(breaks, key=lambda k: k[1])

def vigenere_keyword_break_worker(message, keyword, metric, target_counts, 
                      message_frequency_scaling):
    plaintext = vigenere_decipher(message, keyword)
    counts = message_frequency_scaling(letter_frequencies(plaintext))
    fit = metric(target_counts, counts)
    logger.debug('Vigenere keyword break attempt using key {0} gives fit of '
                 '{1} and decrypt starting: {2}'.format(keyword, 
                     fit, sanitise(plaintext)[:50]))
    return keyword, fit


if __name__ == "__main__":
    import doctest
    doctest.testmod()


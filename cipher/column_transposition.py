import math
import multiprocessing 
from itertools import chain
from support.utilities import *
from support.language_models import *

from logger import logger

def transpositions_of(keyword):
    """Finds the transpostions given by a keyword. For instance, the keyword
    'clever' rearranges to 'celrv', so the first column (0) stays first, the
    second column (1) moves to third, the third column (2) moves to second, 
    and so on.

    If passed a tuple, assume it's already a transposition and just return it.

    >>> transpositions_of('clever')
    (0, 2, 1, 4, 3)
    >>> transpositions_of('fred')
    (3, 2, 0, 1)
    >>> transpositions_of((3, 2, 0, 1))
    (3, 2, 0, 1)
    """
    if isinstance(keyword, tuple):
        return keyword
    else:
        key = deduplicate(keyword)
        transpositions = tuple(key.index(l) for l in sorted(key))
        return transpositions


transpositions = collections.defaultdict(list)
for word in keywords:
    transpositions[transpositions_of(word)] += [word]


def pad(message_len, group_len, fillvalue):
    padding_length = group_len - message_len % group_len
    if padding_length == group_len: padding_length = 0
    padding = ''
    for i in range(padding_length):
        if callable(fillvalue):
            padding += fillvalue()
        else:
            padding += fillvalue
    return padding

def column_transposition_encipher(message, keyword, fillvalue=' ', 
      fillcolumnwise=False,
      emptycolumnwise=False):
    """Enciphers using the column transposition cipher.
    Message is padded to allow all rows to be the same length.

    >>> column_transposition_encipher('hellothere', 'abcdef', fillcolumnwise=True)
    'hlohr eltee '
    >>> column_transposition_encipher('hellothere', 'abcdef', fillcolumnwise=True, emptycolumnwise=True)
    'hellothere  '
    >>> column_transposition_encipher('hellothere', 'abcdef')
    'hellothere  '
    >>> column_transposition_encipher('hellothere', 'abcde')
    'hellothere'
    >>> column_transposition_encipher('hellothere', 'abcde', fillcolumnwise=True, emptycolumnwise=True)
    'hellothere'
    >>> column_transposition_encipher('hellothere', 'abcde', fillcolumnwise=True, emptycolumnwise=False)
    'hlohreltee'
    >>> column_transposition_encipher('hellothere', 'abcde', fillcolumnwise=False, emptycolumnwise=True)
    'htehlelroe'
    >>> column_transposition_encipher('hellothere', 'abcde', fillcolumnwise=False, emptycolumnwise=False)
    'hellothere'
    >>> column_transposition_encipher('hellothere', 'clever', fillcolumnwise=True, emptycolumnwise=True)
    'heotllrehe'
    >>> column_transposition_encipher('hellothere', 'clever', fillcolumnwise=True, emptycolumnwise=False)
    'holrhetlee'
    >>> column_transposition_encipher('hellothere', 'clever', fillcolumnwise=False, emptycolumnwise=True)
    'htleehoelr'
    >>> column_transposition_encipher('hellothere', 'clever', fillcolumnwise=False, emptycolumnwise=False)
    'hleolteher'
    >>> column_transposition_encipher('hellothere', 'cleverly')
    'hleolthre e '
    >>> column_transposition_encipher('hellothere', 'cleverly', fillvalue='!')
    'hleolthre!e!'
    >>> column_transposition_encipher('hellothere', 'cleverly', fillvalue=lambda: '*')
    'hleolthre*e*'
    """
    transpositions = transpositions_of(keyword)
    message += pad(len(message), len(transpositions), fillvalue)
    if fillcolumnwise:
        rows = every_nth(message, len(message) // len(transpositions))
    else:
        rows = chunks(message, len(transpositions))
    transposed = [transpose(r, transpositions) for r in rows]
    if emptycolumnwise:
        return combine_every_nth(transposed)
    else:
        return cat(chain(*transposed))

def column_transposition_decipher(message, keyword, fillvalue=' ', 
      fillcolumnwise=False,
      emptycolumnwise=False):
    """Deciphers using the column transposition cipher.
    Message is padded to allow all rows to be the same length.

    >>> column_transposition_decipher('hellothere', 'abcde', fillcolumnwise=True, emptycolumnwise=True)
    'hellothere'
    >>> column_transposition_decipher('hlohreltee', 'abcde', fillcolumnwise=True, emptycolumnwise=False)
    'hellothere'
    >>> column_transposition_decipher('htehlelroe', 'abcde', fillcolumnwise=False, emptycolumnwise=True)
    'hellothere'
    >>> column_transposition_decipher('hellothere', 'abcde', fillcolumnwise=False, emptycolumnwise=False)
    'hellothere'
    >>> column_transposition_decipher('heotllrehe', 'clever', fillcolumnwise=True, emptycolumnwise=True)
    'hellothere'
    >>> column_transposition_decipher('holrhetlee', 'clever', fillcolumnwise=True, emptycolumnwise=False)
    'hellothere'
    >>> column_transposition_decipher('htleehoelr', 'clever', fillcolumnwise=False, emptycolumnwise=True)
    'hellothere'
    >>> column_transposition_decipher('hleolteher', 'clever', fillcolumnwise=False, emptycolumnwise=False)
    'hellothere'
    """
    transpositions = transpositions_of(keyword)
    message += pad(len(message), len(transpositions), fillvalue)
    if emptycolumnwise:
        rows = every_nth(message, len(message) // len(transpositions))
    else:
        rows = chunks(message, len(transpositions))
    untransposed = [untranspose(r, transpositions) for r in rows]
    if fillcolumnwise:
        return combine_every_nth(untransposed)
    else:
        return cat(chain(*untransposed))

def scytale_encipher(message, rows, fillvalue=' '):
    """Enciphers using the scytale transposition cipher.
    Message is padded with spaces to allow all rows to be the same length.

    >>> scytale_encipher('thequickbrownfox', 3)
    'tcnhkfeboqrxuo iw '
    >>> scytale_encipher('thequickbrownfox', 4)
    'tubnhirfecooqkwx'
    >>> scytale_encipher('thequickbrownfox', 5)
    'tubn hirf ecoo qkwx '
    >>> scytale_encipher('thequickbrownfox', 6)
    'tqcrnxhukof eibwo '
    >>> scytale_encipher('thequickbrownfox', 7)
    'tqcrnx hukof  eibwo  '
    """
    # transpositions = [i for i in range(math.ceil(len(message) / rows))]
    # return column_transposition_encipher(message, transpositions, 
    #     fillvalue=fillvalue, fillcolumnwise=False, emptycolumnwise=True)
    transpositions = [i for i in range(rows)]
    return column_transposition_encipher(message, transpositions, 
        fillvalue=fillvalue, fillcolumnwise=True, emptycolumnwise=False)

def scytale_decipher(message, rows):
    """Deciphers using the scytale transposition cipher.
    Assumes the message is padded so that all rows are the same length.
    
    >>> scytale_decipher('tcnhkfeboqrxuo iw ', 3)
    'thequickbrownfox  '
    >>> scytale_decipher('tubnhirfecooqkwx', 4)
    'thequickbrownfox'
    >>> scytale_decipher('tubn hirf ecoo qkwx ', 5)
    'thequickbrownfox    '
    >>> scytale_decipher('tqcrnxhukof eibwo ', 6)
    'thequickbrownfox  '
    >>> scytale_decipher('tqcrnx hukof  eibwo  ', 7)
    'thequickbrownfox     '
    """
    # transpositions = [i for i in range(math.ceil(len(message) / rows))]
    # return column_transposition_decipher(message, transpositions, 
    #     fillcolumnwise=False, emptycolumnwise=True)
    transpositions = [i for i in range(rows)]
    return column_transposition_decipher(message, transpositions, 
        fillcolumnwise=True, emptycolumnwise=False)


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
    with multiprocessing.Pool() as pool:
        helper_args = [(message, trans, fillcolumnwise, emptycolumnwise,
                        fitness)
                       for trans in translist
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
    with multiprocessing.Pool() as pool:
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

if __name__ == "__main__":
    import doctest
import multiprocessing 
from support.utilities import *
from support.language_models import *
from cipher.keyword_cipher import KeywordWrapAlphabet, keyword_cipher_alphabet_of

from logger import logger

def polybius_grid(keyword, column_order, row_order, letters_to_merge=None,
                  wrap_alphabet=KeywordWrapAlphabet.from_a):
    """Grid for a Polybius cipher, using a keyword to rearrange the
    alphabet.


    >>> polybius_grid('a', 'abcde', 'abcde')['x'] == ('e', 'c')
    True
    >>> polybius_grid('elephant', 'abcde', 'abcde')['e'] == ('a', 'a')
    True
    >>> polybius_grid('elephant', 'abcde', 'abcde')['b'] == ('b', 'c')
    True
    """
    alphabet = keyword_cipher_alphabet_of(keyword, wrap_alphabet=wrap_alphabet)
    if letters_to_merge is None: 
        letters_to_merge = {'j': 'i'}
    grid = {l: k 
            for k, l in zip([(c, r) for c in column_order for r in row_order],
                [l for l in alphabet if l not in letters_to_merge])}
    for l in letters_to_merge:
        grid[l] = grid[letters_to_merge[l]]
    return grid

def polybius_reverse_grid(keyword, column_order, row_order, letters_to_merge=None,
                  wrap_alphabet=KeywordWrapAlphabet.from_a):
    """Grid for decrypting using a Polybius cipher, using a keyword to 
    rearrange the alphabet.

    >>> polybius_reverse_grid('a', 'abcde', 'abcde')['e', 'c'] == 'x'
    True
    >>> polybius_reverse_grid('elephant', 'abcde', 'abcde')['a', 'a'] == 'e'
    True
    >>> polybius_reverse_grid('elephant', 'abcde', 'abcde')['b', 'c'] == 'b'
    True
    """
    alphabet = keyword_cipher_alphabet_of(keyword, wrap_alphabet=wrap_alphabet)
    if letters_to_merge is None: 
        letters_to_merge = {'j': 'i'}
    grid = {k: l 
            for k, l in zip([(c, r) for c in column_order for r in row_order],
                [l for l in alphabet if l not in letters_to_merge])}
    return grid  


def polybius_flatten(pair, column_first):
    """Convert a series of pairs into a single list of characters"""
    if column_first:
        return str(pair[1]) + str(pair[0])
    else:
        return str(pair[0]) + str(pair[1])

def polybius_encipher(message, keyword, column_order, row_order, 
                      column_first=False,
                      letters_to_merge=None, wrap_alphabet=KeywordWrapAlphabet.from_a): 
    """Encipher a message with Polybius cipher, using a keyword to rearrange
    the alphabet


    >>> polybius_encipher('this is a test message for the ' \
          'polybius decipherment', 'elephant', \
          [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], \
          wrap_alphabet=KeywordWrapAlphabet.from_last)
    '2214445544551522115522511155551543114252542214111352123234442355411135441314115451112122'
    >>> polybius_encipher('this is a test message for the ' \
          'polybius decipherment', 'elephant', 'abcde', 'abcde', \
          column_first=False)
    'bbadccddccddaebbaaddbbceaaddddaecbaacadadcbbadaaacdaabedbcccdeddbeaabdccacadaadcceaababb'
    >>> polybius_encipher('this is a test message for the ' \
          'polybius decipherment', 'elephant', 'abcde', 'abcde', \
          column_first=True)
    'bbdaccddccddeabbaaddbbecaaddddeabcaaacadcdbbdaaacaadbadecbccedddebaadbcccadaaacdecaaabbb'
    """
    grid = polybius_grid(keyword, column_order, row_order, letters_to_merge, wrap_alphabet)
    return cat(polybius_flatten(grid[l], column_first)
               for l in message
               if l in grid)


def polybius_decipher(message, keyword, column_order, row_order, 
                      column_first=False,
                      letters_to_merge=None, wrap_alphabet=KeywordWrapAlphabet.from_a):    
    """Decipher a message with a Polybius cipher, using a keyword to rearrange
    the alphabet

    >>> polybius_decipher('bbdaccddccddeabbaaddbbecaaddddeabcaaacadcdbbdaaaca'\
    'adbadecbccedddebaadbcccadaaacdecaaabbb', 'elephant', 'abcde', 'abcde', \
    column_first=False)
    'toisisvtestxessvbephktoefhnugiysweqifoekxelt'

    >>> polybius_decipher('bbdaccddccddeabbaaddbbecaaddddeabcaaacadcdbbdaaaca'\
    'adbadecbccedddebaadbcccadaaacdecaaabbb', 'elephant', 'abcde', 'abcde', \
    column_first=True)
    'thisisatestmessageforthepolybiusdecipherment'
    """
    grid = polybius_reverse_grid(keyword, column_order, row_order, letters_to_merge, wrap_alphabet)
    column_index_type = type(column_order[0])
    row_index_type = type(row_order[0])
    if column_first:
        pairs = [(column_index_type(p[1]), row_index_type(p[0])) for p in chunks(message, 2)]
    else:
        pairs = [(row_index_type(p[0]), column_index_type(p[1])) for p in chunks(message, 2)]
    return cat(grid[p] for p in pairs if p in grid)


def polybius_break_mp(message, column_labels, row_labels,
                      letters_to_merge=None,
                      wordlist=keywords, fitness=Pletters,
                      number_of_solutions=1, chunksize=500):
    """Breaks a Polybius substitution cipher using a dictionary and
    frequency analysis

    >>> polybius_break_mp(polybius_encipher('this is a test message for the ' \
          'polybius decipherment', 'elephant', 'abcde', 'abcde'), \
          'abcde', 'abcde', \
          wordlist=['cat', 'elephant', 'kangaroo']) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    (('elephant', <KeywordWrapAlphabet.from_a: 1>, 'abcde', 'abcde', False), \
    -54.53880...)
    >>> polybius_break_mp(polybius_encipher('this is a test message for the ' \
          'polybius decipherment', 'elephant', 'abcde', 'abcde', column_first=True), \
          'abcde', 'abcde', \
          wordlist=['cat', 'elephant', 'kangaroo']) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    (('elephant', <KeywordWrapAlphabet.from_a: 1>, 'abcde', 'abcde', True), \
    -54.53880...)
    >>> polybius_break_mp(polybius_encipher('this is a test message for the ' \
          'polybius decipherment', 'elephant', 'abcde', 'abcde', column_first=False), \
          'abcde', 'abcde', \
          wordlist=['cat', 'elephant', 'kangaroo']) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    (('elephant', <KeywordWrapAlphabet.from_a: 1>, 'abcde', 'abcde', False), \
    -54.53880...)
    >>> polybius_break_mp(polybius_encipher('this is a test message for the ' \
          'polybius decipherment', 'elephant', 'abcde', 'pqrst', column_first=True), \
          'abcde', 'pqrst', \
          wordlist=['cat', 'elephant', 'kangaroo']) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    (('elephant', <KeywordWrapAlphabet.from_a: 1>, 'abcde', 'pqrst', True), \
    -54.53880...)
    """
    if letters_to_merge is None: 
        letters_to_merge = {'j': 'i'}
    with multiprocessing.Pool() as pool:
        helper_args = [(message, word, wrap, 
                        column_labels, row_labels, column_first, 
                        letters_to_merge, 
                        fitness)
                       for word in wordlist
                       for wrap in KeywordWrapAlphabet
                       for column_first in [False, True]]
        # Gotcha: the helper function here needs to be defined at the top level
        #   (limitation of Pool.starmap)
        breaks = pool.starmap(polybius_break_worker, helper_args, chunksize)
        if number_of_solutions == 1:
            return max(breaks, key=lambda k: k[1])
        else:
            return sorted(breaks, key=lambda k: k[1], reverse=True)[:number_of_solutions]

def polybius_break_worker(message, keyword, wrap_alphabet, 
                          column_order, row_order, column_first, 
                          letters_to_merge, 
                          fitness):
    plaintext = polybius_decipher(message, keyword, 
                                  column_order, row_order, 
                                  column_first=column_first,
                                  letters_to_merge=letters_to_merge, 
                                  wrap_alphabet=wrap_alphabet)
    if plaintext:
        fit = fitness(plaintext)
    else:
        fit = float('-inf')
    logger.debug('Polybius break attempt using key {0} (wrap={1}, merging {2}), '
                 'columns as {3}, rows as {4} (column_first={5}) '
                 'gives fit of {6} and decrypt starting: '
                 '{7}'.format(keyword, wrap_alphabet, letters_to_merge,
                              column_order, row_order, column_first,
                              fit, sanitise(plaintext)[:50]))
    return (keyword, wrap_alphabet, column_order, row_order, column_first), fit

if __name__ == "__main__":
    import doctest
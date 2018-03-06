import multiprocessing 
from support.utilities import *
from support.language_models import *
from cipher.keyword_cipher import KeywordWrapAlphabet, keyword_cipher_alphabet_of

from logger import logger

def bifid_grid(keyword, wrap_alphabet, letter_mapping):
    """Create the grids for a Bifid cipher
    """
    cipher_alphabet = keyword_cipher_alphabet_of(keyword, wrap_alphabet)
    if letter_mapping is None:
        letter_mapping = {'j': 'i'}
    translation = ''.maketrans(letter_mapping)
    cipher_alphabet = cat(collections.OrderedDict.fromkeys(cipher_alphabet.translate(translation)))
    f_grid = {k: ((i // 5) + 1, (i % 5) + 1) 
              for i, k in enumerate(cipher_alphabet)}
    r_grid = {((i // 5) + 1, (i % 5) + 1): k 
              for i, k in enumerate(cipher_alphabet)}
    return translation, f_grid, r_grid

def bifid_encipher(message, keyword, wrap_alphabet=KeywordWrapAlphabet.from_a, 
                   letter_mapping=None, period=None, fillvalue=None):
    """Bifid cipher

    >>> bifid_encipher("indiajelly", 'iguana')
    'ibidonhprm'
    >>> bifid_encipher("indiacurry", 'iguana', period=4)
    'ibnhgaqltm'
    >>> bifid_encipher("indiacurry", 'iguana', period=4, fillvalue='x')
    'ibnhgaqltzml'
    """
    translation, f_grid, r_grid = bifid_grid(keyword, wrap_alphabet, letter_mapping)
    
    t_message = message.translate(translation)
    pairs0 = [f_grid[l] for l in sanitise(t_message)]
    if period:
        chunked_pairs = [pairs0[i:i+period] for i in range(0, len(pairs0), period)]
        if len(chunked_pairs[-1]) < period and fillvalue:
            chunked_pairs[-1] += [f_grid[fillvalue]] * (period - len(chunked_pairs[-1]))
    else:
        chunked_pairs = [pairs0]
    
    pairs1 = []
    for c in chunked_pairs:
        items = sum(list(list(i) for i in zip(*c)), [])
        p = [(items[i], items[i+1]) for i in range(0, len(items), 2)]
        pairs1 += p
    
    return cat(r_grid[p] for p in pairs1)


def bifid_decipher(message, keyword, wrap_alphabet=KeywordWrapAlphabet.from_a, 
                   letter_mapping=None, period=None, fillvalue=None):
    """Decipher with bifid cipher

    >>> bifid_decipher('ibidonhprm', 'iguana')
    'indiaielly'
    >>> bifid_decipher("ibnhgaqltm", 'iguana', period=4)
    'indiacurry'
    >>> bifid_decipher("ibnhgaqltzml", 'iguana', period=4)
    'indiacurryxx'
    """
    translation, f_grid, r_grid = bifid_grid(keyword, wrap_alphabet, letter_mapping)
    
    t_message = message.translate(translation)
    pairs0 = [f_grid[l] for l in sanitise(t_message)]
    if period:
        chunked_pairs = [pairs0[i:i+period] for i in range(0, len(pairs0), period)]
        if len(chunked_pairs[-1]) < period and fillvalue:
            chunked_pairs[-1] += [f_grid[fillvalue]] * (period - len(chunked_pairs[-1]))
    else:
        chunked_pairs = [pairs0]
        
    pairs1 = []
    for c in chunked_pairs:
        items = [j for i in c for j in i]
        gap = len(c)
        p = [(items[i], items[i+gap]) for i in range(gap)]
        pairs1 += p

    return cat(r_grid[p] for p in pairs1) 


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
    with multiprocessing.Pool() as pool:
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

if __name__ == "__main__":
    import doctest
from enum import Enum
from itertools import starmap, cycle
import multiprocessing
from cipher.caesar import *
from support.utilities import *
from support.language_models import *

from logger import logger

def vigenere_encipher(message, keyword):
    """Vigenere encipher

    >>> vigenere_encipher('hello', 'abc')
    'hfnlp'
    """
    shifts = [pos(l) for l in sanitise(keyword)]
    pairs = zip(message, cycle(shifts))
    return cat([caesar_encipher_letter(l, k) for l, k in pairs])

def vigenere_decipher(message, keyword):
    """Vigenere decipher

    >>> vigenere_decipher('hfnlp', 'abc')
    'hello'
    """
    shifts = [pos(l) for l in sanitise(keyword)]
    pairs = zip(message, cycle(shifts))
    return cat([caesar_decipher_letter(l, k) for l, k in pairs])


def beaufort_encipher(message, keyword):
    """Beaufort encipher

    >>> beaufort_encipher('inhisjournaldatedtheidesofoctober', 'arcanaimperii')
    'sevsvrusyrrxfayyxuteemazudmpjmmwr'
    """
    shifts = [pos(l) for l in sanitise(keyword)]
    pairs = zip(message, cycle(shifts))
    return cat([unpos(k - pos(l)) for l, k in pairs])

beaufort_decipher = beaufort_encipher    

beaufort_variant_encipher=vigenere_decipher
beaufort_variant_decipher=vigenere_encipher


def index_of_coincidence_scan(text, max_key_length=20):
    """Finds the index of coincidence of the text, using different chunk sizes."""
    stext = sanitise(text)
    iocs = {}
    for i in range(1, max_key_length + 1):
        splits = every_nth(stext, i)
        mean_ioc = sum(index_of_coincidence(s) for s in splits) / i
        iocs[i] = mean_ioc
    return iocs

def vigenere_keyword_break_mp(message, wordlist=keywords, fitness=Pletters,
                              chunksize=500):
    """Breaks a vigenere cipher using a dictionary and frequency analysis.

    >>> vigenere_keyword_break_mp(vigenere_encipher(sanitise('this is a test ' \
             'message for the vigenere decipherment'), 'cat'), \
             wordlist=['cat', 'elephant', 'kangaroo']) # doctest: +ELLIPSIS
    ('cat', -52.9472712...)
    """
    with multiprocessing.Pool() as pool:
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
    ('florence', -307.5473096...)
    """
    def worker(message, key_length, fitness):
        splits = every_nth(sanitised_message, key_length)
        key = cat([unpos(caesar_break(s)[0]) for s in splits])
        plaintext = vigenere_decipher(message, key)
        fit = fitness(plaintext)
        return key, fit
    sanitised_message = sanitise(message)
    results = starmap(worker, [(sanitised_message, i, fitness)
                               for i in range(1, max_key_length+1)])
    return max(results, key=lambda k: k[1])


def beaufort_sub_break(message, fitness=Pletters):
    """Breaks one chunk of a Beaufort cipher with frequency analysis

    >>> beaufort_sub_break('samwpplggnnmmyaazgympjapopnwiywwomwspgpjmefwmawx' \
      'jafjhxwwwdigxshnlywiamhyshtasxptwueahhytjwsn') # doctest: +ELLIPSIS
    (0, -117.4492...)
    >>> beaufort_sub_break('eyprzjjzznxymrygryjqmqhznjrjjapenejznawngnnezgza' \
      'dgndknaogpdjneadadazlhkhxkryevrronrmdjnndjlo') # doctest: +ELLIPSIS
    (17, -114.9598...)
    """
    best_shift = 0
    best_fit = float('-inf')
    for key in range(26):
        plaintext = [unpos(key - pos(l)) for l in message]
        fit = fitness(plaintext)
        logger.debug('Beaufort sub break attempt using key {0} gives fit of {1} '
                     'and decrypt starting: {2}'.format(key, fit,
                                                        plaintext[:50]))
        if fit > best_fit:
            best_fit = fit
            best_key = key
    logger.info('Beaufort sub break best fit: key {0} gives fit of {1} and '
                'decrypt starting: {2}'.format(best_key, best_fit, 
                    cat([unpos(best_key - pos(l)) for l in message[:50]])))
    return best_key, best_fit


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
        splits = every_nth(message, key_length)
        key = cat([unpos(beaufort_sub_break(s)[0]) for s in splits])
        plaintext = beaufort_decipher(message, key)
        fit = fitness(plaintext)
        return key, fit
    sanitised_message = sanitise(message)
    results = starmap(worker, [(sanitised_message, i, fitness)
                               for i in range(1, max_key_length+1)])
    return max(results, key=lambda k: k[1])    


def beaufort_variant_frequency_break(message, max_key_length=20, fitness=Pletters):
    """Breaks a Beaufort cipher with frequency analysis

    >>> beaufort_variant_frequency_break(beaufort_variant_encipher(sanitise("It is time to " \
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
        key = cat([unpos(-caesar_break(s)[0]) for s in splits])
        plaintext = beaufort_variant_decipher(message, key)
        fit = fitness(plaintext)
        return key, fit
    sanitised_message = sanitise(message)
    results = starmap(worker, [(sanitised_message, i, fitness)
                               for i in range(1, max_key_length+1)])
    return max(results, key=lambda k: k[1])

if __name__ == "__main__":
    import doctest
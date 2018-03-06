from support.utilities import *
from support.language_models import *

from logger import logger

def caesar_encipher_letter(accented_letter, shift):
    """Encipher a letter, given a shift amount

    >>> caesar_encipher_letter('a', 1)
    'b'
    >>> caesar_encipher_letter('a', 2)
    'c'
    >>> caesar_encipher_letter('b', 2)
    'd'
    >>> caesar_encipher_letter('x', 2)
    'z'
    >>> caesar_encipher_letter('y', 2)
    'a'
    >>> caesar_encipher_letter('z', 2)
    'b'
    >>> caesar_encipher_letter('z', -1)
    'y'
    >>> caesar_encipher_letter('a', -1)
    'z'
    >>> caesar_encipher_letter('A', 1)
    'B'
    >>> caesar_encipher_letter('é', 1)
    'f'
    """
    # letter = unaccent(accented_letter)
    # if letter in string.ascii_letters:
    #     if letter in string.ascii_uppercase:
    #         alphabet_start = ord('A')
    #     else:
    #         alphabet_start = ord('a')
    #     return chr(((ord(letter) - alphabet_start + shift) % 26) + 
    #                alphabet_start)
    # else:
    #     return letter

    letter = unaccent(accented_letter)
    if letter in string.ascii_letters:
        cipherletter = unpos(pos(letter) + shift)
        if letter in string.ascii_uppercase:
            return cipherletter.upper()
        else:
            return cipherletter
    else:
        return letter

def caesar_decipher_letter(letter, shift):
    """Decipher a letter, given a shift amount
    
    >>> caesar_decipher_letter('b', 1)
    'a'
    >>> caesar_decipher_letter('b', 2)
    'z'
    """
    return caesar_encipher_letter(letter, -shift)

def caesar_encipher(message, shift):
    """Encipher a message with the Caesar cipher of given shift
    
    >>> caesar_encipher('abc', 1)
    'bcd'
    >>> caesar_encipher('abc', 2)
    'cde'
    >>> caesar_encipher('abcxyz', 2)
    'cdezab'
    >>> caesar_encipher('ab cx yz', 2)
    'cd ez ab'
    >>> caesar_encipher('Héllo World!', 2)
    'Jgnnq Yqtnf!'
    """
    enciphered = [caesar_encipher_letter(l, shift) for l in message]
    return cat(enciphered)

def caesar_decipher(message, shift):
    """Decipher a message with the Caesar cipher of given shift
    
    >>> caesar_decipher('bcd', 1)
    'abc'
    >>> caesar_decipher('cde', 2)
    'abc'
    >>> caesar_decipher('cd ez ab', 2)
    'ab cx yz'
    >>> caesar_decipher('Jgnnq Yqtnf!', 2)
    'Hello World!'
    """
    return caesar_encipher(message, -shift)


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

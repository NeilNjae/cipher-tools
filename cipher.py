import string
import collections
import logging
import math
from itertools import zip_longest, cycle, chain
from language_models import *

logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler('cipher.log'))
logger.setLevel(logging.WARNING)
#logger.setLevel(logging.INFO)
#logger.setLevel(logging.DEBUG)


modular_division_table = [[0]*26 for _ in range(26)]
for a in range(26):
    for b in range(26):
        c = (a * b) % 26
        modular_division_table[b][c] = a


def every_nth(text, n, fillvalue=''):
    """Returns n strings, each of which consists of every nth character, 
    starting with the 0th, 1st, 2nd, ... (n-1)th character
    
    >>> every_nth(string.ascii_lowercase, 5)
    ['afkpuz', 'bglqv', 'chmrw', 'dinsx', 'ejoty']
    >>> every_nth(string.ascii_lowercase, 1)
    ['abcdefghijklmnopqrstuvwxyz']
    >>> every_nth(string.ascii_lowercase, 26) # doctest: +NORMALIZE_WHITESPACE
    ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 
     'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    >>> every_nth(string.ascii_lowercase, 5, fillvalue='!')
    ['afkpuz', 'bglqv!', 'chmrw!', 'dinsx!', 'ejoty!']
    """
    split_text = [text[i:i+n] for i in range(0, len(text), n)]
    return [''.join(l) for l in zip_longest(*split_text, fillvalue=fillvalue)]

def combine_every_nth(split_text):
    """Reforms a text split into every_nth strings
    
    >>> combine_every_nth(every_nth(string.ascii_lowercase, 5))
    'abcdefghijklmnopqrstuvwxyz'
    >>> combine_every_nth(every_nth(string.ascii_lowercase, 1))
    'abcdefghijklmnopqrstuvwxyz'
    >>> combine_every_nth(every_nth(string.ascii_lowercase, 26))
    'abcdefghijklmnopqrstuvwxyz'
    """
    return ''.join([''.join(l) 
                    for l in zip_longest(*split_text, fillvalue='')])

def chunks(text, n, fillvalue=None):
    """Split a text into chunks of n characters

    >>> chunks('abcdefghi', 3)
    ['abc', 'def', 'ghi']
    >>> chunks('abcdefghi', 4)
    ['abcd', 'efgh', 'i']
    >>> chunks('abcdefghi', 4, fillvalue='!')
    ['abcd', 'efgh', 'i!!!']
    """
    if fillvalue:
        padding = fillvalue[0] * (n - len(text) % n)
    else:
        padding = ''
    return [(text+padding)[i:i+n] for i in range(0, len(text), n)]

def transpose(items, transposition):
    """Moves items around according to the given transposition
    
    >>> transpose(['a', 'b', 'c', 'd'], (0,1,2,3))
    ['a', 'b', 'c', 'd']
    >>> transpose(['a', 'b', 'c', 'd'], (3,1,2,0))
    ['d', 'b', 'c', 'a']
    >>> transpose([10,11,12,13,14,15], (3,2,4,1,5,0))
    [13, 12, 14, 11, 15, 10]
    """
    transposed = [''] * len(transposition)
    for p, t in enumerate(transposition):
       transposed[p] = items[t]
    return transposed

def untranspose(items, transposition):
    """Undoes a transpose
    
    >>> untranspose(['a', 'b', 'c', 'd'], [0,1,2,3])
    ['a', 'b', 'c', 'd']
    >>> untranspose(['d', 'b', 'c', 'a'], [3,1,2,0])
    ['a', 'b', 'c', 'd']
    >>> untranspose([13, 12, 14, 11, 15, 10], [3,2,4,1,5,0])
    [10, 11, 12, 13, 14, 15]
    """
    transposed = [''] * len(transposition)
    for p, t in enumerate(transposition):
       transposed[t] = items[p]
    return transposed

def deduplicate(text):
    return list(collections.OrderedDict.fromkeys(text))


def caesar_encipher_letter(letter, shift):
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
    """
    if letter in string.ascii_letters:
        if letter in string.ascii_uppercase:
            alphabet_start = ord('A')
        else:
            alphabet_start = ord('a')
        return chr(((ord(letter) - alphabet_start + shift) % 26) + 
                   alphabet_start)
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
    """
    enciphered = [caesar_encipher_letter(l, shift) for l in message]
    return ''.join(enciphered)

def caesar_decipher(message, shift):
    """Encipher a message with the Caesar cipher of given shift
    
    >>> caesar_decipher('bcd', 1)
    'abc'
    >>> caesar_decipher('cde', 2)
    'abc'
    >>> caesar_decipher('cd ez ab', 2)
    'ab cx yz'
    """
    return caesar_encipher(message, -shift)

def affine_encipher_letter(letter, multiplier=1, adder=0, one_based=True):
    """Encipher a letter, given a multiplier and adder
    
    >>> ''.join([affine_encipher_letter(l, 3, 5, True) \
            for l in string.ascii_uppercase])
    'HKNQTWZCFILORUXADGJMPSVYBE'
    >>> ''.join([affine_encipher_letter(l, 3, 5, False) \
            for l in string.ascii_uppercase])
    'FILORUXADGJMPSVYBEHKNQTWZC'
    """
    if letter in string.ascii_letters:
        if letter in string.ascii_uppercase:
            alphabet_start = ord('A')
        else:
            alphabet_start = ord('a')
        letter_number = ord(letter) - alphabet_start
        if one_based: letter_number += 1
        cipher_number = (letter_number * multiplier + adder) % 26
        if one_based: cipher_number -= 1
        return chr(cipher_number % 26 + alphabet_start)
    else:
        return letter

def affine_decipher_letter(letter, multiplier=1, adder=0, one_based=True):
    """Encipher a letter, given a multiplier and adder
    
    >>> ''.join([affine_decipher_letter(l, 3, 5, True) \
            for l in 'HKNQTWZCFILORUXADGJMPSVYBE'])
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    >>> ''.join([affine_decipher_letter(l, 3, 5, False) \
            for l in 'FILORUXADGJMPSVYBEHKNQTWZC'])
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    """
    if letter in string.ascii_letters:
        if letter in string.ascii_uppercase:
            alphabet_start = ord('A')
        else:
            alphabet_start = ord('a')
        cipher_number = ord(letter) - alphabet_start
        if one_based: cipher_number += 1
        plaintext_number = ( 
            modular_division_table[multiplier]
                                  [(cipher_number - adder) % 26] )
        if one_based: plaintext_number -= 1
        return chr(plaintext_number % 26 + alphabet_start) 
    else:
        return letter

def affine_encipher(message, multiplier=1, adder=0, one_based=True):
    """Encipher a message
    
    >>> affine_encipher('hours passed during which jerico tried every ' \
           'trick he could think of', 15, 22, True)
    'lmyfu bkuusd dyfaxw claol psfaom jfasd snsfg jfaoe ls omytd jlaxe mh'
    """
    enciphered = [affine_encipher_letter(l, multiplier, adder, one_based) 
                  for l in message]
    return ''.join(enciphered)

def affine_decipher(message, multiplier=1, adder=0, one_based=True):
    """Decipher a message
    
    >>> affine_decipher('lmyfu bkuusd dyfaxw claol psfaom jfasd snsfg ' \
           'jfaoe ls omytd jlaxe mh', 15, 22, True)
    'hours passed during which jerico tried every trick he could think of'
    """
    enciphered = [affine_decipher_letter(l, multiplier, adder, one_based) 
                  for l in message]
    return ''.join(enciphered)


def keyword_cipher_alphabet_of(keyword, wrap_alphabet=0):
    """Find the cipher alphabet given a keyword.
    wrap_alphabet controls how the rest of the alphabet is added
    after the keyword.
    0 : from 'a'
    1 : from the last letter in the sanitised keyword
    2 : from the largest letter in the sanitised keyword

    >>> keyword_cipher_alphabet_of('bayes')
    'bayescdfghijklmnopqrtuvwxz'
    >>> keyword_cipher_alphabet_of('bayes', 0)
    'bayescdfghijklmnopqrtuvwxz'
    >>> keyword_cipher_alphabet_of('bayes', 1)
    'bayestuvwxzcdfghijklmnopqr'
    >>> keyword_cipher_alphabet_of('bayes', 2)
    'bayeszcdfghijklmnopqrtuvwx'
    """
    if wrap_alphabet == 0:
        cipher_alphabet = ''.join(deduplicate(sanitise(keyword) + 
                                              string.ascii_lowercase))
    else:
        if wrap_alphabet == 1:
            last_keyword_letter = deduplicate(sanitise(keyword))[-1]
        else:
            last_keyword_letter = sorted(sanitise(keyword))[-1]
        last_keyword_position = string.ascii_lowercase.find(
            last_keyword_letter) + 1
        cipher_alphabet = ''.join(
            deduplicate(sanitise(keyword) + 
                        string.ascii_lowercase[last_keyword_position:] + 
                        string.ascii_lowercase))
    return cipher_alphabet


def keyword_encipher(message, keyword, wrap_alphabet=0):
    """Enciphers a message with a keyword substitution cipher.
    wrap_alphabet controls how the rest of the alphabet is added
    after the keyword.
    0 : from 'a'
    1 : from the last letter in the sanitised keyword
    2 : from the largest letter in the sanitised keyword

    >>> keyword_encipher('test message', 'bayes')
    'rsqr ksqqbds'
    >>> keyword_encipher('test message', 'bayes', 0)
    'rsqr ksqqbds'
    >>> keyword_encipher('test message', 'bayes', 1)
    'lskl dskkbus'
    >>> keyword_encipher('test message', 'bayes', 2)
    'qspq jsppbcs'
    """
    cipher_alphabet = keyword_cipher_alphabet_of(keyword, wrap_alphabet)
    cipher_translation = ''.maketrans(string.ascii_lowercase, cipher_alphabet)
    return message.lower().translate(cipher_translation)

def keyword_decipher(message, keyword, wrap_alphabet=0):
    """Deciphers a message with a keyword substitution cipher.
    wrap_alphabet controls how the rest of the alphabet is added
    after the keyword.
    0 : from 'a'
    1 : from the last letter in the sanitised keyword
    2 : from the largest letter in the sanitised keyword
    
    >>> keyword_decipher('rsqr ksqqbds', 'bayes')
    'test message'
    >>> keyword_decipher('rsqr ksqqbds', 'bayes', 0)
    'test message'
    >>> keyword_decipher('lskl dskkbus', 'bayes', 1)
    'test message'
    >>> keyword_decipher('qspq jsppbcs', 'bayes', 2)                                                                                            
    'test message'
    """
    cipher_alphabet = keyword_cipher_alphabet_of(keyword, wrap_alphabet)
    cipher_translation = ''.maketrans(cipher_alphabet, string.ascii_lowercase)
    return message.lower().translate(cipher_translation)


def vigenere_encipher(message, keyword):
    """Vigenere encipher

    >>> vigenere_encipher('hello', 'abc')
    'hfnlp'
    """
    shifts = [ord(l) - ord('a') for l in sanitise(keyword)]
    pairs = zip(message, cycle(shifts))
    return ''.join([caesar_encipher_letter(l, k) for l, k in pairs])

def vigenere_decipher(message, keyword):
    """Vigenere decipher

    >>> vigenere_decipher('hfnlp', 'abc')
    'hello'
    """
    shifts = [ord(l) - ord('a') for l in sanitise(keyword)]
    pairs = zip(message, cycle(shifts))
    return ''.join([caesar_decipher_letter(l, k) for l, k in pairs])

beaufort_encipher=vigenere_decipher
beaufort_decipher=vigenere_encipher


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
        return ''.join(chain(*transposed))

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
    message += pad(len(message), len(transpositions), '*')
    if emptycolumnwise:
        rows = every_nth(message, len(message) // len(transpositions))
    else:
        rows = chunks(message, len(transpositions))
    untransposed = [untranspose(r, transpositions) for r in rows]
    if fillcolumnwise:
        return combine_every_nth(untransposed)
    else:
        return ''.join(chain(*untransposed))

def scytale_encipher(message, rows, fillvalue=' '):
    """Enciphers using the scytale transposition cipher.
    Message is padded with spaces to allow all rows to be the same length.

    >>> scytale_encipher('thequickbrownfox', 3)
    'tcnhkfeboqrxuo iw '
    >>> scytale_encipher('thequickbrownfox', 4)
    'tubnhirfecooqkwx'
    >>> scytale_encipher('thequickbrownfox', 5)
    'tubnhirfecooqkwx'
    >>> scytale_encipher('thequickbrownfox', 6)
    'tqcrnxhukof eibwo '
    >>> scytale_encipher('thequickbrownfox', 7)
    'tqcrnxhukof eibwo '
    """
    transpositions = [i for i in range(math.ceil(len(message) / rows))]
    return column_transposition_encipher(message, transpositions, 
        fillcolumnwise=False, emptycolumnwise=True)

def scytale_decipher(message, rows):
    """Deciphers using the scytale transposition cipher.
    Assumes the message is padded so that all rows are the same length.
    
    >>> scytale_decipher('tcnhkfeboqrxuo iw ', 3)
    'thequickbrownfox  '
    >>> scytale_decipher('tubnhirfecooqkwx', 4)
    'thequickbrownfox'
    >>> scytale_decipher('tubnhirfecooqkwx', 5)
    'thequickbrownfox'
    >>> scytale_decipher('tqcrnxhukof eibwo ', 6)
    'thequickbrownfox  '
    >>> scytale_decipher('tqcrnxhukof eibwo ', 7)
    'thequickbrownfox  '
    """
    transpositions = [i for i in range(math.ceil(len(message) / rows))]
    return column_transposition_decipher(message, transpositions, 
        fillcolumnwise=False, emptycolumnwise=True)


if __name__ == "__main__":
    import doctest
    doctest.testmod()

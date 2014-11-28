import string
import collections
import math
from enum import Enum
from itertools import zip_longest, cycle, chain
from language_models import *


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
    split_text = chunks(text, n, fillvalue)
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
    letter = unaccent(accented_letter)
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
    >>> caesar_encipher('Héllo World!', 2)
    'Jgnnq Yqtnf!'
    """
    enciphered = [caesar_encipher_letter(l, shift) for l in message]
    return ''.join(enciphered)

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

def affine_encipher_letter(accented_letter, multiplier=1, adder=0, one_based=True):
    """Encipher a letter, given a multiplier and adder
    
    >>> ''.join([affine_encipher_letter(l, 3, 5, True) \
            for l in string.ascii_uppercase])
    'HKNQTWZCFILORUXADGJMPSVYBE'
    >>> ''.join([affine_encipher_letter(l, 3, 5, False) \
            for l in string.ascii_uppercase])
    'FILORUXADGJMPSVYBEHKNQTWZC'
    """
    letter = unaccent(accented_letter)
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
                                  [(cipher_number - adder) % 26])
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


class KeywordWrapAlphabet(Enum):
    from_a = 1
    from_last = 2
    from_largest = 3


def keyword_cipher_alphabet_of(keyword, wrap_alphabet=KeywordWrapAlphabet.from_a):
    """Find the cipher alphabet given a keyword.
    wrap_alphabet controls how the rest of the alphabet is added
    after the keyword.

    >>> keyword_cipher_alphabet_of('bayes')
    'bayescdfghijklmnopqrtuvwxz'
    >>> keyword_cipher_alphabet_of('bayes', KeywordWrapAlphabet.from_a)
    'bayescdfghijklmnopqrtuvwxz'
    >>> keyword_cipher_alphabet_of('bayes', KeywordWrapAlphabet.from_last)
    'bayestuvwxzcdfghijklmnopqr'
    >>> keyword_cipher_alphabet_of('bayes', KeywordWrapAlphabet.from_largest)
    'bayeszcdfghijklmnopqrtuvwx'
    """
    if wrap_alphabet == KeywordWrapAlphabet.from_a:
        cipher_alphabet = ''.join(deduplicate(sanitise(keyword) + 
                                              string.ascii_lowercase))
    else:
        if wrap_alphabet == KeywordWrapAlphabet.from_last:
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


def keyword_encipher(message, keyword, wrap_alphabet=KeywordWrapAlphabet.from_a):
    """Enciphers a message with a keyword substitution cipher.
    wrap_alphabet controls how the rest of the alphabet is added
    after the keyword.
    0 : from 'a'
    1 : from the last letter in the sanitised keyword
    2 : from the largest letter in the sanitised keyword

    >>> keyword_encipher('test message', 'bayes')
    'rsqr ksqqbds'
    >>> keyword_encipher('test message', 'bayes', KeywordWrapAlphabet.from_a)
    'rsqr ksqqbds'
    >>> keyword_encipher('test message', 'bayes', KeywordWrapAlphabet.from_last)
    'lskl dskkbus'
    >>> keyword_encipher('test message', 'bayes', KeywordWrapAlphabet.from_largest)
    'qspq jsppbcs'
    """
    cipher_alphabet = keyword_cipher_alphabet_of(keyword, wrap_alphabet)
    cipher_translation = ''.maketrans(string.ascii_lowercase, cipher_alphabet)
    return unaccent(message).lower().translate(cipher_translation)

def keyword_decipher(message, keyword, wrap_alphabet=KeywordWrapAlphabet.from_a):
    """Deciphers a message with a keyword substitution cipher.
    wrap_alphabet controls how the rest of the alphabet is added
    after the keyword.
    0 : from 'a'
    1 : from the last letter in the sanitised keyword
    2 : from the largest letter in the sanitised keyword
    
    >>> keyword_decipher('rsqr ksqqbds', 'bayes')
    'test message'
    >>> keyword_decipher('rsqr ksqqbds', 'bayes', KeywordWrapAlphabet.from_a)
    'test message'
    >>> keyword_decipher('lskl dskkbus', 'bayes', KeywordWrapAlphabet.from_last)
    'test message'
    >>> keyword_decipher('qspq jsppbcs', 'bayes', KeywordWrapAlphabet.from_largest)
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
    message += pad(len(message), len(transpositions), fillvalue)
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


def railfence_encipher(message, height, fillvalue=''):
    """Railfence cipher.
    Works by splitting the text into sections, then reading across them to
    generate the rows in the cipher. The rows are then combined to form the
    ciphertext.

    Example: the plaintext "hellotherefriends", with a height of four, written 
    out in the railfence as 
       h h i
       etere*
       lorfns
       l e d
    (with the * showing the one character to finish the last section). 
    Each 'section' is two columns, but unfolded. In the example, the first
    section is 'hellot'.

    >>> railfence_encipher('hellothereavastmeheartiesthisisalongpieceoftextfortestingrailfenceciphers', 2, fillvalue='!')
    'hlohraateerishsslnpeefetotsigaleccpeselteevsmhatetiiaogicotxfretnrifneihr!'
    >>> railfence_encipher('hellothereavastmeheartiesthisisalongpieceoftextfortestingrailfenceciphers', 3, fillvalue='!')
    'horaersslpeeosglcpselteevsmhatetiiaogicotxfretnrifneihr!!lhateihsnefttiaece!'
    >>> railfence_encipher('hellothereavastmeheartiesthisisalongpieceoftextfortestingrailfenceciphers', 5, fillvalue='!')
    'hresleogcseeemhetaocofrnrner!!lhateihsnefttiaece!!ltvsatiigitxetifih!!oarspeslp!'
    >>> railfence_encipher('hellothereavastmeheartiesthisisalongpieceoftextfortestingrailfenceciphers', 10, fillvalue='!')
    'hepisehagitnr!!lernesge!!lmtocerh!!otiletap!!tseaorii!!hassfolc!!evtitffe!!rahsetec!!eixn!'
    >>> railfence_encipher('hellothereavastmeheartiesthisisalongpieceoftextfortestingrailfenceciphers', 3)
    'horaersslpeeosglcpselteevsmhatetiiaogicotxfretnrifneihrlhateihsnefttiaece'
    >>> railfence_encipher('hellothereavastmeheartiesthisisalongpieceoftextfortestingrailfenceciphers', 5)
    'hresleogcseeemhetaocofrnrnerlhateihsnefttiaeceltvsatiigitxetifihoarspeslp'
    >>> railfence_encipher('hellothereavastmeheartiesthisisalongpieceoftextfortestingrailfenceciphers', 7)
    'haspolsevsetgifrifrlatihnettaeelemtiocxernhorersleesgcptehaiaottneihesfic'
    """
    sections = chunks(message, (height - 1) * 2, fillvalue=fillvalue)
    n_sections = len(sections)
    # Add the top row
    rows = [''.join([s[0] for s in sections])]
    # process the middle rows of the grid
    for r in range(1, height-1):
        rows += [''.join([s[r:r+1] + s[height*2-r-2:height*2-r-1] for s in sections])]
    # process the bottom row
    rows += [''.join([s[height - 1:height] for s in sections])]
    # rows += [' '.join([s[height - 1] for s in sections])]
    return ''.join(rows)

def railfence_decipher(message, height, fillvalue=''):
    """Railfence decipher. 
    Works by reconstructing the grid used to generate the ciphertext, then
    unfolding the sections so the text can be concatenated together.

    Example: given the ciphertext 'hhieterelorfnsled' and a height of 4, first
    work out that the second row has a character missing, find the rows of the
    grid, then split the section into its two columns.

    'hhieterelorfnsled' is split into
        h h i
        etere
        lorfns
        l e d
    (spaces added for clarity), which is stored in 'rows'. This is then split
    into 'down_rows' and 'up_rows':

    down_rows:
       hhi
       eee
       lrn
       led

    up_rows:
       tr
       ofs

    These are then zipped together (after the up_rows are reversed) to recover 
    the plaintext.

    Most of the procedure is about finding the correct lengths for each row then
    splitting the ciphertext into those rows.

    >>> railfence_decipher('hlohraateerishsslnpeefetotsigaleccpeselteevsmhatetiiaogicotxfretnrifneihr!', 2).strip('!')
    'hellothereavastmeheartiesthisisalongpieceoftextfortestingrailfenceciphers'
    >>> railfence_decipher('horaersslpeeosglcpselteevsmhatetiiaogicotxfretnrifneihr!!lhateihsnefttiaece!', 3).strip('!')
    'hellothereavastmeheartiesthisisalongpieceoftextfortestingrailfenceciphers'
    >>> railfence_decipher('hresleogcseeemhetaocofrnrner!!lhateihsnefttiaece!!ltvsatiigitxetifih!!oarspeslp!', 5).strip('!')
    'hellothereavastmeheartiesthisisalongpieceoftextfortestingrailfenceciphers'
    >>> railfence_decipher('hepisehagitnr!!lernesge!!lmtocerh!!otiletap!!tseaorii!!hassfolc!!evtitffe!!rahsetec!!eixn!', 10).strip('!')
    'hellothereavastmeheartiesthisisalongpieceoftextfortestingrailfenceciphers'
    >>> railfence_decipher('horaersslpeeosglcpselteevsmhatetiiaogicotxfretnrifneihrlhateihsnefttiaece', 3)
    'hellothereavastmeheartiesthisisalongpieceoftextfortestingrailfenceciphers'
    >>> railfence_decipher('hresleogcseeemhetaocofrnrnerlhateihsnefttiaeceltvsatiigitxetifihoarspeslp', 5)
    'hellothereavastmeheartiesthisisalongpieceoftextfortestingrailfenceciphers'
    >>> railfence_decipher('haspolsevsetgifrifrlatihnettaeelemtiocxernhorersleesgcptehaiaottneihesfic', 7)
    'hellothereavastmeheartiesthisisalongpieceoftextfortestingrailfenceciphers'
    """
    # find the number and size of the sections, including how many characters
    #   are missing for a full grid
    n_sections = math.ceil(len(message) / ((height - 1) * 2))
    padding_to_add = n_sections * (height - 1) * 2 - len(message)
    # row_lengths are for the both up rows and down rows
    row_lengths = [n_sections] * (height - 1) * 2
    for i in range((height - 1) * 2 - 1, (height - 1) * 2 - (padding_to_add + 1), -1):
        row_lengths[i] -= 1
    # folded_rows are the combined row lengths in the middle of the railfence
    folded_row_lengths = [row_lengths[0]]
    for i in range(1, height-1):
        folded_row_lengths += [row_lengths[i] + row_lengths[-i]]
    folded_row_lengths += [row_lengths[height - 1]]
    # find the rows that form the railfence grid
    rows = []
    row_start = 0
    for i in folded_row_lengths:
        rows += [message[row_start:row_start + i]]
        row_start += i
    # split the rows into the 'down_rows' (those that form the first column of
    #   a section) and the 'up_rows' (those that ofrm the second column of a 
    #   section).
    down_rows = [rows[0]]
    up_rows = []
    for i in range(1, height-1):
        down_rows += [''.join([c for n, c in enumerate(rows[i]) if n % 2 == 0])]
        up_rows += [''.join([c for n, c in enumerate(rows[i]) if n % 2 == 1])]
    down_rows += [rows[-1]]
    up_rows.reverse()
    return ''.join(c for r in zip_longest(*(down_rows + up_rows), fillvalue='') for c in r)


class PocketEnigma(object):
    """A pocket enigma machine
    The wheel is internally represented as a 26-element list self.wheel_map, 
    where wheel_map[i] == j shows that the position i places on from the arrow 
    maps to the position j places on.
    """
    def __init__(self, wheel=1, position='a'):
        """initialise the pocket enigma, including which wheel to use and the
        starting position of the wheel.

        The wheel is either 1 or 2 (the predefined wheels) or a list of letter
        pairs.

        The position is the letter pointed to by the arrow on the wheel.

        >>> pe.wheel_map
        [25, 4, 23, 10, 1, 7, 9, 5, 12, 6, 3, 17, 8, 14, 13, 21, 19, 11, 20, 16, 18, 15, 24, 2, 22, 0]
        >>> pe.position
        0
        """
        self.wheel1 = [('a', 'z'), ('b', 'e'), ('c', 'x'), ('d', 'k'), 
            ('f', 'h'), ('g', 'j'), ('i', 'm'), ('l', 'r'), ('n', 'o'), 
            ('p', 'v'), ('q', 't'), ('s', 'u'), ('w', 'y')]
        self.wheel2 = [('a', 'c'), ('b', 'd'), ('e', 'w'), ('f', 'i'), 
            ('g', 'p'), ('h', 'm'), ('j', 'k'), ('l', 'n'), ('o', 'q'), 
            ('r', 'z'), ('s', 'u'), ('t', 'v'), ('x', 'y')]
        if wheel == 1:
            self.make_wheel_map(self.wheel1)
        elif wheel == 2:
            self.make_wheel_map(self.wheel2)
        else:
            self.validate_wheel_spec(wheel)
            self.make_wheel_map(wheel)
        if position in string.ascii_lowercase:
            self.position = ord(position) - ord('a')
        else:
            self.position = position

    def make_wheel_map(self, wheel_spec):
        """Expands a wheel specification from a list of letter-letter pairs
        into a full wheel_map.

        >>> pe.make_wheel_map(pe.wheel2)
        [2, 3, 0, 1, 22, 8, 15, 12, 5, 10, 9, 13, 7, 11, 16, 6, 14, 25, 20, 21, 18, 19, 4, 24, 23, 17]
        """
        self.validate_wheel_spec(wheel_spec)
        self.wheel_map = [0] * 26
        for p in wheel_spec:
            self.wheel_map[ord(p[0]) - ord('a')] = ord(p[1]) - ord('a')
            self.wheel_map[ord(p[1]) - ord('a')] = ord(p[0]) - ord('a')
        return self.wheel_map

    def validate_wheel_spec(self, wheel_spec):
        """Validates that a wheel specificaiton will turn into a valid wheel
        map.

        >>> pe.validate_wheel_spec([])
        Traceback (most recent call last):
            ...
        ValueError: Wheel specification has 0 pairs, requires 13
        >>> pe.validate_wheel_spec([('a', 'b', 'c')]*13)
        Traceback (most recent call last):
            ...
        ValueError: Not all mappings in wheel specificationhave two elements
        >>> pe.validate_wheel_spec([('a', 'b')]*13)
        Traceback (most recent call last):
            ...
        ValueError: Wheel specification does not contain 26 letters
        """
        if len(wheel_spec) != 13:
            raise ValueError("Wheel specification has {} pairs, requires 13".
                format(len(wheel_spec)))
        for p in wheel_spec:
            if len(p) != 2:
                raise ValueError("Not all mappings in wheel specification"
                    "have two elements")
        if len(set([p[0] for p in wheel_spec] + 
                    [p[1] for p in wheel_spec])) != 26:
            raise ValueError("Wheel specification does not contain 26 letters")

    def encipher_letter(self, letter):
        """Enciphers a single letter, by advancing the wheel before looking up
        the letter on the wheel.

        >>> pe.set_position('f')
        5
        >>> pe.encipher_letter('k')
        'h'
        """
        self.advance()
        return self.lookup(letter)
    decipher_letter = encipher_letter

    def lookup(self, letter):
        """Look up what a letter enciphers to, without turning the wheel.

        >>> pe.set_position('f')
        5
        >>> ''.join([pe.lookup(l) for l in string.ascii_lowercase])
        'udhbfejcpgmokrliwntsayqzvx'
        >>> pe.lookup('A')
        ''
        """
        if letter in string.ascii_lowercase:
            return chr(
                (self.wheel_map[(ord(letter) - ord('a') - self.position) % 26] + 
                    self.position) % 26 + 
                ord('a'))
        else:
            return ''

    def advance(self):
        """Advances the wheel one position.

        >>> pe.set_position('f')
        5
        >>> pe.advance()
        6
        """
        self.position = (self.position + 1) % 26
        return self.position

    def encipher(self, message, starting_position=None):
        """Enciphers a whole message.

        >>> pe.set_position('f')
        5
        >>> pe.encipher('helloworld')
        'kjsglcjoqc'
        >>> pe.set_position('f')
        5
        >>> pe.encipher('kjsglcjoqc')
        'helloworld'
        >>> pe.encipher('helloworld', starting_position = 'x')
        'egrekthnnf'
        """
        if starting_position:
            self.set_position(starting_position)
        transformed = ''
        for l in message:
            transformed += self.encipher_letter(l)
        return transformed
    decipher = encipher

    def set_position(self, position):
        """Sets the position of the wheel, by specifying the letter the arrow
        points to.

        >>> pe.set_position('a')
        0
        >>> pe.set_position('m')
        12
        >>> pe.set_position('z')
        25
        """
        self.position = ord(position) - ord('a')
        return self.position


if __name__ == "__main__":
    import doctest
    doctest.testmod(extraglobs={'pe': PocketEnigma(1, 'a')})

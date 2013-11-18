import string
import collections
import norms
import logging
import math
from itertools import zip_longest, repeat
from segment import segment
from multiprocessing import Pool

# To time a run:
#
# import timeit
# c5a = open('2012/5a.ciphertext', 'r').read()
# timeit.timeit('keyword_break(c5a)', setup='gc.enable() ; from __main__ import c5a ; from cipher import keyword_break', number=1)
# timeit.repeat('keyword_break_mp(c5a, chunksize=500)', setup='gc.enable() ; from __main__ import c5a ; from cipher import keyword_break_mp', repeat=5, number=1

logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler('cipher.log'))
logger.setLevel(logging.WARNING)
#logger.setLevel(logging.INFO)
#logger.setLevel(logging.DEBUG)

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

with open('words.txt', 'r') as f:
    keywords = [line.rstrip() for line in f]

modular_division_table = [[0]*26 for x in range(26)]
for a in range(26):
    for b in range(26):
        c = (a * b) % 26
        modular_division_table[b][c] = a


def sanitise(text):
    """Remove all non-alphabetic characters and convert the text to lowercase
    
    >>> sanitise('The Quick')
    'thequick'
    >>> sanitise('The Quick BROWN fox jumped! over... the (9lazy) DOG')
    'thequickbrownfoxjumpedoverthelazydog'
    """
    sanitised = [c.lower() for c in text if c in string.ascii_letters]
    return ''.join(sanitised)

def ngrams(text, n):
    """Returns all n-grams of a text
    
    >>> ngrams(sanitise('the quick brown fox'), 2) # doctest: +NORMALIZE_WHITESPACE
    ['th', 'he', 'eq', 'qu', 'ui', 'ic', 'ck', 'kb', 'br', 'ro', 'ow', 'wn', 
     'nf', 'fo', 'ox']
    >>> ngrams(sanitise('the quick brown fox'), 4) # doctest: +NORMALIZE_WHITESPACE
    ['theq', 'hequ', 'equi', 'quic', 'uick', 'ickb', 'ckbr', 'kbro', 'brow', 
     'rown', 'ownf', 'wnfo', 'nfox']
    """
    return [text[i:i+n] for i in range(len(text)-n+1)]

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

def transpose(items, transposition):
    """Moves items around according to the given transposition
    
    >>> transpose(['a', 'b', 'c', 'd'], [0,1,2,3])
    ['a', 'b', 'c', 'd']
    >>> transpose(['a', 'b', 'c', 'd'], [3,1,2,0])
    ['d', 'b', 'c', 'a']
    >>> transpose([10,11,12,13,14,15], [3,2,4,1,5,0])	
    [13, 12, 14, 11, 15, 10]
    """
    transposed = list(repeat('', len(transposition)))
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
    transposed  = list(repeat('', len(transposition)))
    for p, t in enumerate(transposition):
       transposed[t] = items[p]
    return transposed


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
    """
    counts = collections.defaultdict(int)
    for c in text: 
        counts[c] += 1
    return counts
letter_frequencies = frequencies

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
        plaintext_number = ( modular_division_table[multiplier]
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

def scytale_encipher(message, rows):
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
    if len(message) % rows != 0:
        message += ' '*(rows - len(message) % rows)
    row_length = round(len(message) / rows)
    slices = [message[i:i+row_length] 
              for i in range(0, len(message), row_length)]
    return ''.join([''.join(r) for r in zip_longest(*slices, fillvalue='')])

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
    cols = round(len(message) / rows)
    columns = [message[i:i+rows] for i in range(0, cols * rows, rows)]
    return ''.join([''.join(c) for c in zip_longest(*columns, fillvalue='')])


def transpositions_of(keyword):
    """
    >>> transpositions_of('clever')
    [0, 2, 1, 4, 3]
    """
    transpositions = []
    key = deduplicate(keyword)
    for l in sorted(key):
        transpositions += [key.index(l)]
    return transpositions

def column_transposition_encipher(message, keyword):
    """
    >>> column_transposition_encipher('hellothere', 'clever')
    'hleolteher'
    """
    transpositions = transpositions_of(keyword)
    columns = every_nth(message, len(transpositions), fillvalue=' ')
    transposed_columns = transpose(columns, transpositions)
    return combine_every_nth(transposed_columns)

def column_transposition_decipher(message, keyword):
    """
    >>> column_transposition_decipher('hleolteher', 'clever')
    'hellothere'
    """
    transpositions = transpositions_of(keyword)
    columns = every_nth(message, len(transpositions), fillvalue=' ')
    transposed_columns = untranspose(columns, transpositions)
    return combine_every_nth(transposed_columns)



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
        breaks = pool.starmap(keyword_break_one, helper_args, chunksize) 
        return min(breaks, key=lambda k: k[1])

def keyword_break_one(message, keyword, wrap_alphabet, metric, target_counts, 
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
    for key in range(1, 20):
        if len(message) % key == 0:
            plaintext = scytale_decipher(message, key)
            counts = message_frequency_scaling(frequencies(
                         ngrams(sanitise(plaintext), 2)))
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


if __name__ == "__main__":
    import doctest
    doctest.testmod()

import string
import collections
import norms
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler('cipher.log'))
logger.setLevel(logging.WARNING)
#logger.setLevel(logging.INFO)

english_counts = collections.defaultdict(int)
with open('count_1l.txt', 'r') as f:
    for line in f:
        (letter, count) = line.split("\t")
        english_counts[letter] = int(count)
normalised_english_counts = norms.normalise(english_counts)        

keywords = []
with open('words.txt', 'r') as f:
    keywords = [line.rstrip() for line in f]


modular_division_table = [[0]*26 for x in range(26)]
for a in range(26):
    for b in range(26):
        c = (a * b) % 26
        modular_division_table[b][c] = a

modular_division_table_one_based = [[0]*27 for x in range(27)]
for a in range(27):
    for b in range(27):
        c = ((a * b)-1) % 26 + 1
        modular_division_table_one_based[b][c] = a



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
    
    >>> ngrams(sanitise('the quick brown fox'), 2)
    [('t', 'h'), ('h', 'e'), ('e', 'q'), ('q', 'u'), ('u', 'i'), ('i', 'c'), ('c', 'k'), ('k', 'b'), ('b', 'r'), ('r', 'o'), ('o', 'w'), ('w', 'n'), ('n', 'f'), ('f', 'o'), ('o', 'x')]
    >>> ngrams(sanitise('the quick brown fox'), 4)
    [('t', 'h', 'e', 'q'), ('h', 'e', 'q', 'u'), ('e', 'q', 'u', 'i'), ('q', 'u', 'i', 'c'), ('u', 'i', 'c', 'k'), ('i', 'c', 'k', 'b'), ('c', 'k', 'b', 'r'), ('k', 'b', 'r', 'o'), ('b', 'r', 'o', 'w'), ('r', 'o', 'w', 'n'), ('o', 'w', 'n', 'f'), ('w', 'n', 'f', 'o'), ('n', 'f', 'o', 'x')]
    """
    return [tuple(text[i:i+n]) for i in range(len(text)-n+1)]

def letter_frequencies(text):
    """Count the number of occurrences of each character in text
    
    >>> sorted(letter_frequencies('abcdefabc').items())
    [('a', 2), ('b', 2), ('c', 2), ('d', 1), ('e', 1), ('f', 1)]
    >>> sorted(letter_frequencies('the quick brown fox jumped over the lazy dog').items())
    [(' ', 8), ('a', 1), ('b', 1), ('c', 1), ('d', 2), ('e', 4), ('f', 1), ('g', 1), ('h', 2), ('i', 1), ('j', 1), ('k', 1), ('l', 1), ('m', 1), ('n', 1), ('o', 4), ('p', 1), ('q', 1), ('r', 2), ('t', 2), ('u', 2), ('v', 1), ('w', 1), ('x', 1), ('y', 1), ('z', 1)]
    >>> sorted(letter_frequencies('The Quick BROWN fox jumped! over... the (9lazy) DOG').items())
    [(' ', 8), ('!', 1), ('(', 1), (')', 1), ('.', 3), ('9', 1), ('B', 1), ('D', 1), ('G', 1), ('N', 1), ('O', 2), ('Q', 1), ('R', 1), ('T', 1), ('W', 1), ('a', 1), ('c', 1), ('d', 1), ('e', 4), ('f', 1), ('h', 2), ('i', 1), ('j', 1), ('k', 1), ('l', 1), ('m', 1), ('o', 2), ('p', 1), ('r', 1), ('t', 1), ('u', 2), ('v', 1), ('x', 1), ('y', 1), ('z', 1)]
    >>> sorted(letter_frequencies(sanitise('The Quick BROWN fox jumped! over... the (9lazy) DOG')).items())
    [('a', 1), ('b', 1), ('c', 1), ('d', 2), ('e', 4), ('f', 1), ('g', 1), ('h', 2), ('i', 1), ('j', 1), ('k', 1), ('l', 1), ('m', 1), ('n', 1), ('o', 4), ('p', 1), ('q', 1), ('r', 2), ('t', 2), ('u', 2), ('v', 1), ('w', 1), ('x', 1), ('y', 1), ('z', 1)]
    """
    counts = collections.defaultdict(int)
    for c in text: 
        counts[c] += 1
    return counts

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
        return chr(((ord(letter) - alphabet_start + shift) % 26) + alphabet_start)
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
    
    >>> ''.join([affine_encipher_letter(l, 3, 5, True) for l in string.ascii_uppercase])
    'HKNQTWZCFILORUXADGJMPSVYBE'
    >>> ''.join([affine_encipher_letter(l, 3, 5, False) for l in string.ascii_uppercase])
    'FILORUXADGJMPSVYBEHKNQTWZC'
    """
    if letter in string.ascii_letters:
        if letter in string.ascii_uppercase:
            alphabet_start = ord('A')
        else:
            alphabet_start = ord('a')
        letter_number = ord(letter) - alphabet_start
        if one_based: letter_number += 1
        raw_cipher_number = (letter_number * multiplier + adder)
        cipher_number = 0
        if one_based: 
            cipher_number = (raw_cipher_number - 1) % 26
        else:
            cipher_number = raw_cipher_number % 26        
        return chr(cipher_number + alphabet_start)
    else:
        return letter

def affine_decipher_letter(letter, multiplier=1, adder=0, one_based=True):
    """Encipher a letter, given a multiplier and adder
    
    >>> ''.join([affine_decipher_letter(l, 3, 5, True) for l in 'HKNQTWZCFILORUXADGJMPSVYBE'])
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    >>> ''.join([affine_decipher_letter(l, 3, 5, False) for l in 'FILORUXADGJMPSVYBEHKNQTWZC'])
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    """
    if letter in string.ascii_letters:
        if letter in string.ascii_uppercase:
            alphabet_start = ord('A')
        else:
            alphabet_start = ord('a')
        cipher_number = ord(letter) - alphabet_start
        if one_based: cipher_number += 1
        plaintext_number = 0
        if one_based:
            plaintext_number = (modular_division_table_one_based[multiplier][(cipher_number - adder + 26) % 26] - 1) % 26
        else:
            #plaintext_number = (modular_division_table[multiplier][cipher_number] - adder) % 26
            plaintext_number = modular_division_table[multiplier][(cipher_number - adder + 26) % 26]            
        return chr(plaintext_number + alphabet_start)
    else:
        return letter

def affine_encipher(message, multiplier=1, adder=0, one_based=True):
    """Encipher a message
    
    >>> affine_encipher('hours passed during which jerico tried every trick he could think of', 15, 22, True)
    'lmyfu bkuusd dyfaxw claol psfaom jfasd snsfg jfaoe ls omytd jlaxe mh'
    """
    
    enciphered = [affine_encipher_letter(l, multiplier, adder, one_based) for l in message]
    return ''.join(enciphered)

def affine_decipher(message, multiplier=1, adder=0, one_based=True):
    """Decipher a message
    
    >>> affine_decipher('lmyfu bkuusd dyfaxw claol psfaom jfasd snsfg jfaoe ls omytd jlaxe mh', 15, 22, True)
    'hours passed during which jerico tried every trick he could think of'
    """
    enciphered = [affine_decipher_letter(l, multiplier, adder, one_based) for l in message]
    return ''.join(enciphered)


def keyword_encipher(message, keyword):
    cipher_alphabet = ''.join(deduplicate(sanitise(keyword) + string.ascii_lowercase))
    cipher_translation = ''.maketrans(string.ascii_lowercase, cipher_alphabet)
    return message.lower().translate(cipher_translation)

def keyword_decipher(message, keyword):
    cipher_alphabet = ''.join(deduplicate(sanitise(keyword) + string.ascii_lowercase))
    cipher_translation = ''.maketrans(cipher_alphabet, string.ascii_lowercase)
    return message.lower().translate(cipher_translation)


def caesar_break(message, metric=norms.euclidean_distance, target_frequencies=normalised_english_counts, message_frequency_scaling=norms.normalise):
    """Breaks a Caesar cipher using frequency analysis
    
    >>> caesar_break('ibxcsyorsaqcheyklxivoexlevmrimwxsfiqevvmihrsasrxliwyrhecjsppsamrkwleppfmergefifvmhixscsymjcsyqeoixlm')
    (4, 0.3186395289018361)
    >>> caesar_break('wxwmaxdgheetgwuxztgptedbgznitgwwhpguxyhkxbmhvvtlbhgteeraxlmhiixweblmxgxwmhmaxybkbgztgwztsxwbgmxgmert')
    (19, 0.4215290123583277)
    >>> caesar_break('yltbbqnqnzvguvaxurorgenafsbezqvagbnornfgsbevpnaabjurersvaquvzyvxrnznazlybequrvfohgriraabjtbaruraprur')
    (13, 0.31602920807545154)
    """
    sanitised_message = sanitise(message)
    best_shift = 0
    best_fit = float("inf")
    for shift in range(26):
        plaintext = caesar_decipher(sanitised_message, shift)
        frequencies = message_frequency_scaling(letter_frequencies(plaintext))
        fit = metric(target_frequencies, frequencies)
        logger.info('Caesar break attempt using key {0} gives fit of {1} and decrypt starting: {2}'.format(shift, fit, plaintext[:50]))
        if fit < best_fit:
            best_fit = fit
            best_shift = shift
    logger.info('Caesar break best fit: key {0} gives fit of {1} and decrypt starting: {2}'.format(best_shift, best_fit, caesar_decipher(sanitised_message, best_shift)[:50]))
    return best_shift, best_fit

def affine_break(message, metric=norms.euclidean_distance, target_frequencies=normalised_english_counts, message_frequency_scaling=norms.normalise):
    """Breaks an affine cipher using frequency analysis
    
    >>> affine_break('lmyfu bkuusd dyfaxw claol psfaom jfasd snsfg jfaoe ls omytd jlaxe mh jm bfmibj umis hfsul axubafkjamx. ls kffkxwsd jls ofgbjmwfkiu olfmxmtmwaokttg jlsx ls kffkxwsd jlsi zg tsxwjl. jlsx ls umfjsd jlsi zg hfsqysxog. ls dmmdtsd mx jls bats mh bkbsf. ls bfmctsd kfmyxd jls lyj, mztanamyu xmc jm clm cku tmmeaxw kj lai kxd clm ckuxj.')
    ((15, 22, True), 0.2357036181865554)
    """
    sanitised_message = sanitise(message)
    best_multiplier = 0
    best_adder = 0
    best_one_based = True
    best_fit = float("inf")
    for one_based in [True, False]:
        for multiplier in range(1, 26, 2):
            for adder in range(26):
                plaintext = affine_decipher(sanitised_message, multiplier, adder, one_based)
                frequencies = message_frequency_scaling(letter_frequencies(plaintext))
                fit = metric(target_frequencies, frequencies)
                logger.info('Affine break attempt using key {0}x+{1} ({2}) gives fit of {3} and decrypt starting: {4}'.format(multiplier, adder, one_based, fit, plaintext[:50]))
                if fit < best_fit:
                    best_fit = fit
                    best_multiplier = multiplier
                    best_adder = adder
                    best_one_based = one_based
    logger.info('Affine break best fit with key {0}x+{1} ({2}) gives fit of {3} and decrypt starting: {4}'.format(best_multiplier, best_adder, best_one_based, best_fit, affine_decipher(sanitised_message, best_multiplier, best_adder, best_one_based)[:50]))
    return (best_multiplier, best_adder, best_one_based), best_fit


def keyword_break(message, metric=norms.euclidean_distance, target_frequencies=normalised_english_counts, message_frequency_scaling=norms.normalise):
    best_keyword = ''
    best_fit = float("inf")
    for keyword in keywords:
        plaintext = keyword_decipher(message, keyword)
        frequencies = message_frequency_scaling(letter_frequencies(plaintext))
        fit = metric(target_frequencies, frequencies)
        logger.info('Keyword break attempt using key {0} gives fit of {1} and decrypt starting: {2}'.format(keyword, fit, plaintext[:50]))
        if fit < best_fit:
            best_fit = fit
            best_keyword = keyword
    logger.info('Keyword break best fit with key {0} gives fit of {1} and decrypt starting: {2}'.format(best_keyword, best_fit, keyword_decipher(message, best_keyword)[:50]))
    return best_keyword, best_fit


if __name__ == "__main__":
    import doctest
    doctest.testmod()

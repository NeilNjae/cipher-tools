import string
import collections
import norms

english_counts = collections.defaultdict(int)
with open('count_1l.txt', 'r') as f:
    for line in f:
        (letter, count) = line.split("\t")
        english_counts[letter] = int(count)
normalised_english_counts = norms.normalise(english_counts)        


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

def caesar_break(message, metric=norms.euclidean_distance, target_frequencies=normalised_english_counts, message_frequency_scaling=norms.normalise):
    """Breaks a Caesar cipher using frequency analysis
    
    >>> caesar_break('ibxcsyorsaqcheyklxivoexlevmrimwxsfiqevvmihrsasrxliwyrhecjsppsamrkwleppfmergefifvmhixscsymjcsyqeoixlm')
    (4, 0.3186395289018361)
    >>> caesar_break('jhzhuhfrqilqhgwrdevwudfwuhdvrqlqjwkhqkdylqjvxemhfwhgwrfulwlflvpwkhhasodqdwlrqrisrzhuwkdwmxulglfdovfl')
    (3, 0.3290204286173084)
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
        if fit < best_fit:
            best_fit = fit
            best_shift = shift
    return best_shift, best_fit


if __name__ == "__main__":
    import doctest
    doctest.testmod()

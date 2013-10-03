import string
import collections


def sanitise(text):
    """Remove all non-alphabetic characters and convert the text to lowercase
    
    >>> sanitise('The Quick')
    'thequick'
    >>> sanitise('The Quick BROWN fox jumped! over... the (9lazy) DOG')
    'thequickbrownfoxjumpedoverthelazydog'
    """
    sanitised = [c.lower() for c in text if c in string.ascii_letters]
    return ''.join(sanitised)

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


def normalise_frequencies(frequencies):
    """Scale a set of letter frequenies so they add to 1
    
    >>> sorted(normalise_frequencies(letter_frequencies('abcdefabc')).items())
    [('a', 0.2222222222222222), ('b', 0.2222222222222222), ('c', 0.2222222222222222), ('d', 0.1111111111111111), ('e', 0.1111111111111111), ('f', 0.1111111111111111)]
    >>> sorted(normalise_frequencies(letter_frequencies('the quick brown fox jumped over the lazy dog')).items())
    [(' ', 0.18181818181818182), ('a', 0.022727272727272728), ('b', 0.022727272727272728), ('c', 0.022727272727272728), ('d', 0.045454545454545456), ('e', 0.09090909090909091), ('f', 0.022727272727272728), ('g', 0.022727272727272728), ('h', 0.045454545454545456), ('i', 0.022727272727272728), ('j', 0.022727272727272728), ('k', 0.022727272727272728), ('l', 0.022727272727272728), ('m', 0.022727272727272728), ('n', 0.022727272727272728), ('o', 0.09090909090909091), ('p', 0.022727272727272728), ('q', 0.022727272727272728), ('r', 0.045454545454545456), ('t', 0.045454545454545456), ('u', 0.045454545454545456), ('v', 0.022727272727272728), ('w', 0.022727272727272728), ('x', 0.022727272727272728), ('y', 0.022727272727272728), ('z', 0.022727272727272728)]
    >>> sorted(normalise_frequencies(letter_frequencies('The Quick BROWN fox jumped! over... the (9lazy) DOG')).items())
    [(' ', 0.1568627450980392), ('!', 0.0196078431372549), ('(', 0.0196078431372549), (')', 0.0196078431372549), ('.', 0.058823529411764705), ('9', 0.0196078431372549), ('B', 0.0196078431372549), ('D', 0.0196078431372549), ('G', 0.0196078431372549), ('N', 0.0196078431372549), ('O', 0.0392156862745098), ('Q', 0.0196078431372549), ('R', 0.0196078431372549), ('T', 0.0196078431372549), ('W', 0.0196078431372549), ('a', 0.0196078431372549), ('c', 0.0196078431372549), ('d', 0.0196078431372549), ('e', 0.0784313725490196), ('f', 0.0196078431372549), ('h', 0.0392156862745098), ('i', 0.0196078431372549), ('j', 0.0196078431372549), ('k', 0.0196078431372549), ('l', 0.0196078431372549), ('m', 0.0196078431372549), ('o', 0.0392156862745098), ('p', 0.0196078431372549), ('r', 0.0196078431372549), ('t', 0.0196078431372549), ('u', 0.0392156862745098), ('v', 0.0196078431372549), ('x', 0.0196078431372549), ('y', 0.0196078431372549), ('z', 0.0196078431372549)]
    >>> sorted(normalise_frequencies(letter_frequencies(sanitise('The Quick BROWN fox jumped! over... the (9lazy) DOG'))).items())
    [('a', 0.027777777777777776), ('b', 0.027777777777777776), ('c', 0.027777777777777776), ('d', 0.05555555555555555), ('e', 0.1111111111111111), ('f', 0.027777777777777776), ('g', 0.027777777777777776), ('h', 0.05555555555555555), ('i', 0.027777777777777776), ('j', 0.027777777777777776), ('k', 0.027777777777777776), ('l', 0.027777777777777776), ('m', 0.027777777777777776), ('n', 0.027777777777777776), ('o', 0.1111111111111111), ('p', 0.027777777777777776), ('q', 0.027777777777777776), ('r', 0.05555555555555555), ('t', 0.05555555555555555), ('u', 0.05555555555555555), ('v', 0.027777777777777776), ('w', 0.027777777777777776), ('x', 0.027777777777777776), ('y', 0.027777777777777776), ('z', 0.027777777777777776)]
    """
    total = sum(frequencies.values())
    return dict((k, v / total) for (k, v) in frequencies.items())

def l2_norm(frequencies1, frequencies2):
    """Finds the distances between two frequency profiles, expressed as dictionaries.
    Assumes every key in frequencies1 is also in frequencies2
    
    >>> l2_norm({'a':1, 'b':1, 'c':1}, {'a':1, 'b':1, 'c':1})
    0.0
    >>> l2_norm({'a':2, 'b':2, 'c':2}, {'a':1, 'b':1, 'c':1})
    0.0
    >>> l2_norm({'a':0, 'b':2, 'c':0}, {'a':1, 'b':1, 'c':1})
    0.816496580927726
    >>> l2_norm({'a':0, 'b':1}, {'a':1, 'b':1})
    0.7071067811865476
    """
    f1n = normalise_frequencies(frequencies1)
    f2n = normalise_frequencies(frequencies2)
    total = 0
    for k in f1n.keys():
        total += (f1n[k] - f2n[k]) ** 2
    return total ** 0.5
euclidean_distance = l2_norm

def l1_norm(frequencies1, frequencies2):
    """Finds the distances between two frequency profiles, expressed as dictionaries.
    Assumes every key in frequencies1 is also in frequencies2

    >>> l1_norm({'a':1, 'b':1, 'c':1}, {'a':1, 'b':1, 'c':1})
    0.0
    >>> l1_norm({'a':2, 'b':2, 'c':2}, {'a':1, 'b':1, 'c':1})
    0.0
    >>> l1_norm({'a':0, 'b':2, 'c':0}, {'a':1, 'b':1, 'c':1})
    1.3333333333333333
    >>> l1_norm({'a':0, 'b':1}, {'a':1, 'b':1})
    1.0
    """
    f1n = normalise_frequencies(frequencies1)
    f2n = normalise_frequencies(frequencies2)
    total = 0
    for k in f1n.keys():
        total += abs(f1n[k] - f2n[k])
    return total

def l3_norm(frequencies1, frequencies2):
    """Finds the distances between two frequency profiles, expressed as dictionaries.
    Assumes every key in frequencies1 is also in frequencies2

    >>> l3_norm({'a':1, 'b':1, 'c':1}, {'a':1, 'b':1, 'c':1})
    0.0
    >>> l3_norm({'a':2, 'b':2, 'c':2}, {'a':1, 'b':1, 'c':1})
    0.0
    >>> l3_norm({'a':0, 'b':2, 'c':0}, {'a':1, 'b':1, 'c':1})
    0.7181448966772946
    >>> l3_norm({'a':0, 'b':1}, {'a':1, 'b':1})
    0.6299605249474366
    """
    f1n = normalise_frequencies(frequencies1)
    f2n = normalise_frequencies(frequencies2)
    total = 0
    for k in f1n.keys():
        total += abs(f1n[k] - f2n[k]) ** 3
    return total ** (1/3)

def cosine_distance(frequencies1, frequencies2):
    """Finds the distances between two frequency profiles, expressed as dictionaries.
    Assumes every key in frequencies1 is also in frequencies2

    >>> cosine_distance({'a':1, 'b':1, 'c':1}, {'a':1, 'b':1, 'c':1})
    -2.220446049250313e-16
    >>> cosine_distance({'a':2, 'b':2, 'c':2}, {'a':1, 'b':1, 'c':1})
    -2.220446049250313e-16
    >>> cosine_distance({'a':0, 'b':2, 'c':0}, {'a':1, 'b':1, 'c':1})
    0.42264973081037416
    >>> cosine_distance({'a':0, 'b':1}, {'a':1, 'b':1})
    0.29289321881345254
    """
    numerator = 0
    length1 = 0
    length2 = 0
    for k in frequencies1.keys():
        numerator += frequencies1[k] * frequencies2[k]
        length1 += frequencies1[k]**2
    for k in frequencies2.keys():
        length2 += frequencies2[k]
    return 1 - (numerator / (length1 ** 0.5 * length2 ** 0.5))




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

def caesar_break(message, metric=euclidean_distance):
    sanitised_message = sanitise(message)
    best_shift = 0
    best_fit = float("inf")
    for shift in range(1, 25):
        plaintext = caesar_decipher(sanitised_message, shift)
        frequencies = letter_frequencies(plaintext)
        fit = metric(english_counts, frequencies)
        if fit < best_fit:
            best_fit = fit
            best_shift = shift
    return best_shift, best_fit





english_counts = collections.defaultdict(int)
with open('count_1l.txt', 'r') as f:
    for line in f:
        (letter, count) = line.split("\t")
        english_counts[letter] = int(count)


if __name__ == "__main__":
    import doctest
    doctest.testmod()

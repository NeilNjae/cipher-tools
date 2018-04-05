import string
import collections
import unicodedata
from itertools import zip_longest

# join a a list of letters into a string
cat = ''.join

# join a list of words into a string, separated by spaces
wcat = ' '.join

# join a list of lines, separated by newline
lcat = '\n'.join

def pos(letter): 
    """Return the position of a letter in the alphabet (0-25)"""
    if letter in string.ascii_lowercase:
        return ord(letter) - ord('a')
    elif letter in string.ascii_uppercase:
        return ord(letter) - ord('A')
    else:
        raise ValueError('pos requires input of {} to be an ascii letter'.format(letter))
    
def unpos(number): 
    """Return the letter in the given position in the alphabet (mod 26)"""
    return chr(number % 26 + ord('a'))

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
    return [cat(l) for l in zip_longest(*split_text, fillvalue=fillvalue)]

def combine_every_nth(split_text):
    """Reforms a text split into every_nth strings
    
    >>> combine_every_nth(every_nth(string.ascii_lowercase, 5))
    'abcdefghijklmnopqrstuvwxyz'
    >>> combine_every_nth(every_nth(string.ascii_lowercase, 1))
    'abcdefghijklmnopqrstuvwxyz'
    >>> combine_every_nth(every_nth(string.ascii_lowercase, 26))
    'abcdefghijklmnopqrstuvwxyz'
    """
    return cat([cat(l) 
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


def letters(text):
    """Remove all non-alphabetic characters from a text
    >>> letters('The Quick')
    'TheQuick'
    >>> letters('The Quick BROWN fox jumped! over... the (9lazy) DOG')
    'TheQuickBROWNfoxjumpedoverthelazyDOG'
    """
    return ''.join([c for c in text if c in string.ascii_letters])

# Special characters for conversion, such as smart quotes.
unaccent_specials = ''.maketrans({"’": "'", '“': '"', '”': '"'})

def unaccent(text):
    """Remove all accents from letters. 
    It does this by converting the unicode string to decomposed compatability
    form, dropping all the combining accents, then re-encoding the bytes.

    >>> unaccent('hello')
    'hello'
    >>> unaccent('HELLO')
    'HELLO'
    >>> unaccent('héllo')
    'hello'
    >>> unaccent('héllö')
    'hello'
    >>> unaccent('HÉLLÖ')
    'HELLO'
    """
    translated_text = text.translate(unaccent_specials)
    return unicodedata.normalize('NFKD', translated_text).\
        encode('ascii', 'ignore').\
        decode('utf-8')

def sanitise(text):
    """Remove all non-alphabetic characters and convert the text to lowercase
    
    >>> sanitise('The Quick')
    'thequick'
    >>> sanitise('The Quick BROWN fox jumped! over... the (9lazy) DOG')
    'thequickbrownfoxjumpedoverthelazydog'
    >>> sanitise('HÉLLÖ')
    'hello'
    """
    return letters(unaccent(text)).lower()


def index_of_coincidence(text):
    stext = sanitise(text)
    counts = collections.Counter(stext)
    denom = len(stext) * (len(text) - 1) / 26
    return (
        sum(max(counts[l] * counts[l] - 1, 0) for l in string.ascii_lowercase)
        /
        denom
    )


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
    >>> sorted(frequencies(sanitise('The Quick BROWN fox jumped! over... '\
         'the (9lazy) DOG')).items()) # doctest: +NORMALIZE_WHITESPACE
    [('a', 1), ('b', 1), ('c', 1), ('d', 2), ('e', 4), ('f', 1), ('g', 1),
     ('h', 2), ('i', 1), ('j', 1), ('k', 1), ('l', 1), ('m', 1), ('n', 1),
     ('o', 4), ('p', 1), ('q', 1), ('r', 2), ('t', 2), ('u', 2), ('v', 1),
     ('w', 1), ('x', 1), ('y', 1), ('z', 1)]
    >>> frequencies('abcdefabcdef')['x']
    0
    """
    return collections.Counter(c for c in text)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
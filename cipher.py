import string
import collections
import math
from enum import Enum
from itertools import zip_longest, cycle, chain, count
import numpy as np
from numpy import matrix
from numpy import linalg
from language_models import *
import pprint



from utilities import *
from segment import *

from caesar import *
from affine import *
from keyword import *
from polybius import *
from column_transposition import *
from railfence import *


def make_cadenus_keycolumn(doubled_letters = 'vw', start='a', reverse=False):
    """Makes the key column for a Cadenus cipher (the column down between the
        rows of letters)

    >>> make_cadenus_keycolumn()['a']
    0
    >>> make_cadenus_keycolumn()['b']
    1
    >>> make_cadenus_keycolumn()['c']
    2
    >>> make_cadenus_keycolumn()['v']
    21
    >>> make_cadenus_keycolumn()['w']
    21
    >>> make_cadenus_keycolumn()['z']
    24
    >>> make_cadenus_keycolumn(doubled_letters='ij', start='b', reverse=True)['a']
    1
    >>> make_cadenus_keycolumn(doubled_letters='ij', start='b', reverse=True)['b']
    0
    >>> make_cadenus_keycolumn(doubled_letters='ij', start='b', reverse=True)['c']
    24
    >>> make_cadenus_keycolumn(doubled_letters='ij', start='b', reverse=True)['i']
    18
    >>> make_cadenus_keycolumn(doubled_letters='ij', start='b', reverse=True)['j']
    18
    >>> make_cadenus_keycolumn(doubled_letters='ij', start='b', reverse=True)['v']
    6
    >>> make_cadenus_keycolumn(doubled_letters='ij', start='b', reverse=True)['z']
    2
    """
    index_to_remove = string.ascii_lowercase.find(doubled_letters[0])
    short_alphabet = string.ascii_lowercase[:index_to_remove] + string.ascii_lowercase[index_to_remove+1:]
    if reverse:
        short_alphabet = cat(reversed(short_alphabet))
    start_pos = short_alphabet.find(start)
    rotated_alphabet = short_alphabet[start_pos:] + short_alphabet[:start_pos]
    keycolumn = {l: i for i, l in enumerate(rotated_alphabet)}
    keycolumn[doubled_letters[0]] = keycolumn[doubled_letters[1]]
    return keycolumn

def cadenus_encipher(message, keyword, keycolumn, fillvalue='a'):
    """Encipher with the Cadenus cipher

    >>> cadenus_encipher(sanitise('Whoever has made a voyage up the Hudson ' \
                                  'must remember the Kaatskill mountains. ' \
                                  'They are a dismembered branch of the great'), \
                'wink', \
                make_cadenus_keycolumn(doubled_letters='vw', start='a', reverse=True))
    'antodeleeeuhrsidrbhmhdrrhnimefmthgeaetakseomehetyaasuvoyegrastmmuuaeenabbtpchehtarorikswosmvaleatned'
    >>> cadenus_encipher(sanitise('a severe limitation on the usefulness of ' \
                                  'the cadenus is that every message must be ' \
                                  'a multiple of twenty-five letters long'), \
                'easy', \
                make_cadenus_keycolumn(doubled_letters='vw', start='a', reverse=True))
    'systretomtattlusoatleeesfiyheasdfnmschbhneuvsnpmtofarenuseieeieltarlmentieetogevesitfaisltngeeuvowul'
    """
    rows = chunks(message, len(message) // 25, fillvalue=fillvalue)
    columns = zip(*rows)
    rotated_columns = [col[start:] + col[:start] for start, col in zip([keycolumn[l] for l in keyword], columns)]    
    rotated_rows = zip(*rotated_columns)
    transpositions = transpositions_of(keyword)
    transposed = [transpose(r, transpositions) for r in rotated_rows]
    return cat(chain(*transposed))

def cadenus_decipher(message, keyword, keycolumn, fillvalue='a'):
    """
    >>> cadenus_decipher('antodeleeeuhrsidrbhmhdrrhnimefmthgeaetakseomehetyaa' \
                         'suvoyegrastmmuuaeenabbtpchehtarorikswosmvaleatned', \
                 'wink', \
                 make_cadenus_keycolumn(reverse=True))
    'whoeverhasmadeavoyageupthehudsonmustrememberthekaatskillmountainstheyareadismemberedbranchofthegreat'
    >>> cadenus_decipher('systretomtattlusoatleeesfiyheasdfnmschbhneuvsnpmtof' \
                        'arenuseieeieltarlmentieetogevesitfaisltngeeuvowul', \
                 'easy', \
                 make_cadenus_keycolumn(reverse=True))
    'aseverelimitationontheusefulnessofthecadenusisthateverymessagemustbeamultipleoftwentyfiveletterslong'
    """
    rows = chunks(message, len(message) // 25, fillvalue=fillvalue)
    transpositions = transpositions_of(keyword)
    untransposed_rows = [untranspose(r, transpositions) for r in rows]
    columns = zip(*untransposed_rows)
    rotated_columns = [col[-start:] + col[:-start] for start, col in zip([keycolumn[l] for l in keyword], columns)]    
    rotated_rows = zip(*rotated_columns)
    # return rotated_columns
    return cat(chain(*rotated_rows))


def hill_encipher(matrix, message_letters, fillvalue='a'):
    """Hill cipher

    >>> hill_encipher(np.matrix([[7,8], [11,11]]), 'hellothere')
    'drjiqzdrvx'
    >>> hill_encipher(np.matrix([[6, 24, 1], [13, 16, 10], [20, 17, 15]]), \
        'hello there')
    'tfjflpznvyac'
    """
    n = len(matrix)
    sanitised_message = sanitise(message_letters)
    if len(sanitised_message) % n != 0:
        padding = fillvalue[0] * (n - len(sanitised_message) % n)
    else:
        padding = ''
    message = [pos(c) for c in sanitised_message + padding]
    message_chunks = [message[i:i+n] for i in range(0, len(message), n)]
    # message_chunks = chunks(message, len(matrix), fillvalue=None)
    enciphered_chunks = [((matrix * np.matrix(c).T).T).tolist()[0] 
            for c in message_chunks]
    return cat([unpos(round(l))
            for l in sum(enciphered_chunks, [])])

def hill_decipher(matrix, message, fillvalue='a'):
    """Hill cipher

    >>> hill_decipher(np.matrix([[7,8], [11,11]]), 'drjiqzdrvx')
    'hellothere'
    >>> hill_decipher(np.matrix([[6, 24, 1], [13, 16, 10], [20, 17, 15]]), \
        'tfjflpznvyac')
    'hellothereaa'
    """
    adjoint = linalg.det(matrix)*linalg.inv(matrix)
    inverse_determinant = modular_division_table[int(round(linalg.det(matrix))) % 26][1]
    inverse_matrix = (inverse_determinant * adjoint) % 26
    return hill_encipher(inverse_matrix, message, fillvalue)          


# Where each piece of text ends up in the AMSCO transpositon cipher.
# 'index' shows where the slice appears in the plaintext, with the slice
# from 'start' to 'end'
AmscoSlice = collections.namedtuple('AmscoSlice', ['index', 'start', 'end'])

class AmscoFillStyle(Enum):
    continuous = 1
    same_each_row = 2
    reverse_each_row = 3

def amsco_transposition_positions(message, keyword, 
      fillpattern=(1, 2),
      fillstyle=AmscoFillStyle.continuous,
      fillcolumnwise=False,
      emptycolumnwise=True):
    """Creates the grid for the AMSCO transposition cipher. Each element in the
    grid shows the index of that slice and the start and end positions of the
    plaintext that go to make it up.

    >>> amsco_transposition_positions(string.ascii_lowercase, 'freddy', \
        fillpattern=(1, 2)) # doctest:  +NORMALIZE_WHITESPACE
    [[AmscoSlice(index=3, start=4, end=6),
     AmscoSlice(index=2, start=3, end=4),
     AmscoSlice(index=0, start=0, end=1),
     AmscoSlice(index=1, start=1, end=3),
     AmscoSlice(index=4, start=6, end=7)],
    [AmscoSlice(index=8, start=12, end=13),
     AmscoSlice(index=7, start=10, end=12),
     AmscoSlice(index=5, start=7, end=9),
     AmscoSlice(index=6, start=9, end=10),
     AmscoSlice(index=9, start=13, end=15)],
    [AmscoSlice(index=13, start=19, end=21),
     AmscoSlice(index=12, start=18, end=19),
     AmscoSlice(index=10, start=15, end=16),
     AmscoSlice(index=11, start=16, end=18),
     AmscoSlice(index=14, start=21, end=22)],
    [AmscoSlice(index=18, start=27, end=28),
     AmscoSlice(index=17, start=25, end=27),
     AmscoSlice(index=15, start=22, end=24),
     AmscoSlice(index=16, start=24, end=25),
     AmscoSlice(index=19, start=28, end=30)]]
    """
    transpositions = transpositions_of(keyword)
    fill_iterator = cycle(fillpattern)
    indices = count()
    message_length = len(message)

    current_position = 0
    grid = []
    current_fillpattern = fillpattern
    while current_position < message_length:
        row = []
        if fillstyle == AmscoFillStyle.same_each_row:
            fill_iterator = cycle(fillpattern)
        if fillstyle == AmscoFillStyle.reverse_each_row:
            fill_iterator = cycle(current_fillpattern)
        for _ in range(len(transpositions)):
            index = next(indices)
            gap = next(fill_iterator)
            row += [AmscoSlice(index, current_position, current_position + gap)]
            current_position += gap
        grid += [row]
        if fillstyle == AmscoFillStyle.reverse_each_row:
            current_fillpattern = list(reversed(current_fillpattern))
    return [transpose(r, transpositions) for r in grid]

def amsco_transposition_encipher(message, keyword, 
    fillpattern=(1,2), fillstyle=AmscoFillStyle.reverse_each_row):
    """AMSCO transposition encipher.

    >>> amsco_transposition_encipher('hellothere', 'abc', fillpattern=(1, 2))
    'hoteelhler'
    >>> amsco_transposition_encipher('hellothere', 'abc', fillpattern=(2, 1))
    'hetelhelor'
    >>> amsco_transposition_encipher('hellothere', 'acb', fillpattern=(1, 2))
    'hotelerelh'
    >>> amsco_transposition_encipher('hellothere', 'acb', fillpattern=(2, 1))
    'hetelorlhe'
    >>> amsco_transposition_encipher('hereissometexttoencipher', 'encode')
    'etecstthhomoerereenisxip'
    >>> amsco_transposition_encipher('hereissometexttoencipher', 'cipher', fillpattern=(1, 2))
    'hetcsoeisterereipexthomn'
    >>> amsco_transposition_encipher('hereissometexttoencipher', 'cipher', fillpattern=(1, 2), fillstyle=AmscoFillStyle.continuous)
    'hecsoisttererteipexhomen'
    >>> amsco_transposition_encipher('hereissometexttoencipher', 'cipher', fillpattern=(2, 1))
    'heecisoosttrrtepeixhemen'
    >>> amsco_transposition_encipher('hereissometexttoencipher', 'cipher', fillpattern=(1, 3, 2))
    'hxtomephescieretoeisnter'
    >>> amsco_transposition_encipher('hereissometexttoencipher', 'cipher', fillpattern=(1, 3, 2), fillstyle=AmscoFillStyle.continuous)
    'hxomeiphscerettoisenteer'
    """
    grid = amsco_transposition_positions(message, keyword, 
        fillpattern=fillpattern, fillstyle=fillstyle)
    ct_as_grid = [[message[s.start:s.end] for s in r] for r in grid]
    return combine_every_nth(ct_as_grid)


def amsco_transposition_decipher(message, keyword, 
    fillpattern=(1,2), fillstyle=AmscoFillStyle.reverse_each_row):
    """AMSCO transposition decipher

    >>> amsco_transposition_decipher('hoteelhler', 'abc', fillpattern=(1, 2))
    'hellothere'
    >>> amsco_transposition_decipher('hetelhelor', 'abc', fillpattern=(2, 1))
    'hellothere'
    >>> amsco_transposition_decipher('hotelerelh', 'acb', fillpattern=(1, 2))
    'hellothere'
    >>> amsco_transposition_decipher('hetelorlhe', 'acb', fillpattern=(2, 1))
    'hellothere'
    >>> amsco_transposition_decipher('etecstthhomoerereenisxip', 'encode')
    'hereissometexttoencipher'
    >>> amsco_transposition_decipher('hetcsoeisterereipexthomn', 'cipher', fillpattern=(1, 2))
    'hereissometexttoencipher'
    >>> amsco_transposition_decipher('hecsoisttererteipexhomen', 'cipher', fillpattern=(1, 2), fillstyle=AmscoFillStyle.continuous)
    'hereissometexttoencipher'
    >>> amsco_transposition_decipher('heecisoosttrrtepeixhemen', 'cipher', fillpattern=(2, 1))
    'hereissometexttoencipher'
    >>> amsco_transposition_decipher('hxtomephescieretoeisnter', 'cipher', fillpattern=(1, 3, 2))
    'hereissometexttoencipher'
    >>> amsco_transposition_decipher('hxomeiphscerettoisenteer', 'cipher', fillpattern=(1, 3, 2), fillstyle=AmscoFillStyle.continuous)
    'hereissometexttoencipher'
    """

    grid = amsco_transposition_positions(message, keyword, 
        fillpattern=fillpattern, fillstyle=fillstyle)
    transposed_sections = [s for c in [l for l in zip(*grid)] for s in c]
    plaintext_list = [''] * len(transposed_sections)
    current_pos = 0
    for slice in transposed_sections:
        plaintext_list[slice.index] = message[current_pos:current_pos-slice.start+slice.end][:len(message[slice.start:slice.end])]
        current_pos += len(message[slice.start:slice.end])
    return cat(plaintext_list)


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


def autokey_encipher(message, keyword):
    """Encipher with the autokey cipher

    >>> autokey_encipher('meetatthefountain', 'kilt')
    'wmpmmxxaeyhbryoca'
    """
    shifts = [pos(l) for l in keyword + message]
    pairs = zip(message, shifts)
    return cat([caesar_encipher_letter(l, k) for l, k in pairs])

def autokey_decipher(ciphertext, keyword):
    """Decipher with the autokey cipher

    >>> autokey_decipher('wmpmmxxaeyhbryoca', 'kilt')
    'meetatthefountain'
    """
    plaintext = []
    keys = list(keyword)
    for c in ciphertext:
        plaintext_letter = caesar_decipher_letter(c, pos(keys[0]))
        plaintext += [plaintext_letter]
        keys = keys[1:] + [plaintext_letter]
    return cat(plaintext)


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
            self.position = pos(position)
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
            self.wheel_map[pos(p[0])] = pos(p[1])
            self.wheel_map[pos(p[1])] = pos(p[0])
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
        >>> cat([pe.lookup(l) for l in string.ascii_lowercase])
        'udhbfejcpgmokrliwntsayqzvx'
        >>> pe.lookup('A')
        ''
        """
        if letter in string.ascii_lowercase:
            return unpos(
                (self.wheel_map[(pos(letter) - self.position) % 26] + 
                    self.position))
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
        self.position = pos(position)
        return self.position


if __name__ == "__main__":
    import doctest
    doctest.testmod(extraglobs={'pe': PocketEnigma(1, 'a')})

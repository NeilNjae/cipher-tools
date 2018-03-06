from support.utilities import *
from support.language_models import *

from logger import logger


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


def pocket_enigma_break_by_crib(message, wheel_spec, crib, crib_position):
    """Break a pocket enigma using a crib (some plaintext that's expected to
    be in a certain position). Returns a list of possible starting wheel
    positions that could produce the crib.

    >>> pocket_enigma_break_by_crib('kzpjlzmoga', 1, 'h', 0)
    ['a', 'f', 'q']
    >>> pocket_enigma_break_by_crib('kzpjlzmoga', 1, 'he', 0)
    ['a']
    >>> pocket_enigma_break_by_crib('kzpjlzmoga', 1, 'll', 2)
    ['a']
    >>> pocket_enigma_break_by_crib('kzpjlzmoga', 1, 'l', 2)
    ['a']
    >>> pocket_enigma_break_by_crib('kzpjlzmoga', 1, 'l', 3)
    ['a', 'j', 'n']
    >>> pocket_enigma_break_by_crib('aaaaa', 1, 'l', 3)
    []
    """
    pe = PocketEnigma(wheel=wheel_spec)
    possible_positions = []
    for p in string.ascii_lowercase:
        pe.set_position(p)
        plaintext = pe.decipher(message)
        if plaintext[crib_position:crib_position+len(crib)] == crib:
            possible_positions += [p]
    return possible_positions

if __name__ == "__main__":
    import doctest
    doctest.testmod(extraglobs={'pe': PocketEnigma(1, 'a')})
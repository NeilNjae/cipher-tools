
# coding: utf-8

##################################
# # Enigma machine
##################################
# Specification from [Codes and Ciphers](http://www.codesandciphers.org.uk/enigma/rotorspec.htm) page.
# 
# Example Enigma machines from [Louise Dale](http://enigma.louisedade.co.uk/enigma.html) (full simulation) and [EnigmaCo](http://enigmaco.de/enigma/enigma.html) (good animation of the wheels, but no ring settings).
# 
# There's also the nice Enigma simulator for Android by [Franklin Heath](https://franklinheath.co.uk/2012/02/04/our-first-app-published-enigma-simulator/), available on the [Google Play store](https://play.google.com/store/apps/details?id=uk.co.franklinheath.enigmasim&hl=en_GB).



import string
import collections
import multiprocessing
import itertools

# Some convenience functions

cat = ''.join

def clean(text): return cat(l.lower() for l in text if l in string.ascii_letters)

def pos(letter): 
    if letter in string.ascii_lowercase:
        return ord(letter) - ord('a')
    elif letter in string.ascii_uppercase:
        return ord(letter) - ord('A')
    else:
        return ''
    
def unpos(number): return chr(number % 26 + ord('a'))


wheel_i_spec = 'ekmflgdqvzntowyhxuspaibrcj'
wheel_ii_spec = 'ajdksiruxblhwtmcqgznpyfvoe'
wheel_iii_spec = 'bdfhjlcprtxvznyeiwgakmusqo'
wheel_iv_spec = 'esovpzjayquirhxlnftgkdcmwb'
wheel_v_spec = 'vzbrgityupsdnhlxawmjqofeck'
wheel_vi_spec = 'jpgvoumfyqbenhzrdkasxlictw'
wheel_vii_spec = 'nzjhgrcxmyswboufaivlpekqdt'
wheel_viii_spec = 'fkqhtlxocbjspdzramewniuygv'
beta_wheel_spec = 'leyjvcnixwpbqmdrtakzgfuhos'
gamma_wheel_spec = 'fsokanuerhmbtiycwlqpzxvgjd'

wheel_i_pegs = ['q']
wheel_ii_pegs = ['e']
wheel_iii_pegs = ['v']
wheel_iv_pegs = ['j']
wheel_v_pegs = ['z']
wheel_vi_pegs = ['z', 'm']
wheel_vii_pegs = ['z', 'm']
wheel_viii_pegs = ['z', 'm']

reflector_b_spec = 'ay br cu dh eq fs gl ip jx kn mo tz vw'
reflector_c_spec = 'af bv cp dj ei go hy kr lz mx nw tq su'



class LetterTransformer(object):
    """A generic substitution cipher, that has different transforms in the 
    forward and backward directions. It requires that the transforms for all
    letters by provided.

    >>> lt = LetterTransformer([('z', 'a')] + [(l, string.ascii_lowercase[i+1]) \
            for i, l in enumerate(string.ascii_lowercase[:-1])], \
            raw_transform = True)
    >>> lt.forward_map
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 0]
    >>> lt.backward_map
    [25, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

    >>> lt = LetterTransformer(cat(collections.OrderedDict.fromkeys('zyxwc' + string.ascii_lowercase)))
    >>> lt.forward_map
    [25, 24, 23, 22, 2, 0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    >>> lt.backward_map
    [5, 6, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 3, 2, 1, 0]
    >>> cat(lt.forward(l) for l in string.ascii_lowercase)
    'zyxwcabdefghijklmnopqrstuv'
    >>> cat(lt.backward(l) for l in string.ascii_lowercase)
    'fgehijklmnopqrstuvwxyzdcba'
    """
    def __init__(self, specification, raw_transform=False):
        if raw_transform:
            transform = specification
        else:
            transform = self.parse_specification(specification)
        self.validate_transform(transform)
        self.make_transform_map(transform)
    
    def parse_specification(self, specification):
        return list(zip(string.ascii_lowercase, clean(specification)))
        # return specification
    
    def validate_transform(self, transform):
        """A set of pairs, of from-to"""
        if len(transform) != 26:
            raise ValueError("Transform specification has {} pairs, requires 26".
                format(len(transform)))
        for p in transform:
            if len(p) != 2:
                raise ValueError("Not all mappings in transform "
                    "have two elements")
        if len(set([p[0] for p in transform])) != 26:
            raise ValueError("Transform specification must list 26 origin letters") 
        if len(set([p[1] for p in transform])) != 26:
            raise ValueError("Transform specification must list 26 destination letters") 

    def make_empty_transform(self):
        self.forward_map = [0] * 26
        self.backward_map = [0] * 26
            
    def make_transform_map(self, transform):
        self.make_empty_transform()
        for p in transform:
            self.forward_map[pos(p[0])] = pos(p[1])
            self.backward_map[pos(p[1])] = pos(p[0])
        return self.forward_map, self.backward_map
    
    def forward(self, letter):
        if letter in string.ascii_lowercase:
            return unpos(self.forward_map[pos(letter)])
        else:
            return ''
                
    def backward(self, letter):
        if letter in string.ascii_lowercase:
            return unpos(self.backward_map[pos(letter)])
        else:
            return ''


class Plugboard(LetterTransformer):
    """A plugboard, a type of letter transformer where forward and backward
    transforms are the same. If a letter isn't explicitly transformed, it is 
    kept as it is.

    >>> pb = Plugboard('ua pf rq so ni ey bg hl tx zj'.upper())
    >>> pb.forward_map
    [20, 6, 2, 3, 24, 15, 1, 11, 13, 25, 10, 7, 12, 8, 18, 5, 17, 16, 14, 23, 0, 21, 22, 19, 4, 9]
    >>> pb.forward_map == pb.backward_map
    True
    >>> cat(pb.forward(l) for l in string.ascii_lowercase)
    'ugcdypblnzkhmisfrqoxavwtej'
    >>> cat(pb.backward(l) for l in string.ascii_lowercase)
    'ugcdypblnzkhmisfrqoxavwtej'
    """
    def parse_specification(self, specification):
        return [tuple(clean(p)) for p in specification.split()]
    
    def validate_transform(self, transform):
        """A set of pairs, of from-to"""
        for p in transform:
            if len(p) != 2:
                raise ValueError("Not all mappings in transform"
                    "have two elements")
    
    def make_empty_transform(self):
        self.forward_map = list(range(26))
        self.backward_map = list(range(26))
        
    def make_transform_map(self, transform):
        expanded_transform = transform + [tuple(reversed(p)) for p in transform]
        return super(Plugboard, self).make_transform_map(expanded_transform)




class Reflector(Plugboard):
    """A reflector is a plugboard that requires 13 transforms.

    >>> reflector_b = Reflector(reflector_b_spec)
    >>> reflector_b.forward_map == reflector_b.backward_map
    True
    >>> reflector_b.forward_map
    [24, 17, 20, 7, 16, 18, 11, 3, 15, 23, 13, 6, 14, 10, 12, 8, 4, 1, 5, 25, 2, 22, 21, 9, 0, 19]
    >>> cat(reflector_b.forward(l) for l in string.ascii_lowercase)
    'yruhqsldpxngokmiebfzcwvjat'
    >>> cat(reflector_b.backward(l) for l in string.ascii_lowercase)
    'yruhqsldpxngokmiebfzcwvjat'
    """
    def validate_transform(self, transform):
        if len(transform) != 13:
            raise ValueError("Reflector specification has {} pairs, requires 13".
                format(len(transform)))
        if len(set([p[0] for p in transform] + 
                    [p[1] for p in transform])) != 26:
            raise ValueError("Reflector specification does not contain 26 letters")
        try:
            super(Reflector, self).validate_transform(transform)
        except ValueError as v:
            raise ValueError("Not all mappings in reflector have two elements")




class SimpleWheel(LetterTransformer):
    """A wheel is a transform that rotates.

    Looking from the right, letters go in sequence a-b-c clockwise around the 
    wheel. 

    The position of the wheel is the number of spaces anticlockwise the wheel
    has turned.

    Letter inputs and outputs are given relative to the frame holding the wheel,
    so if the wheel is advanced three places, an input of 'p' will enter the 
    wheel on the position under the wheel's 'q' label.

    >>> rotor_1_transform = list(zip(string.ascii_lowercase, 'EKMFLGDQVZNTOWYHXUSPAIBRCJ'.lower()))
    >>> wheel_1 = SimpleWheel(rotor_1_transform, raw_transform=True)
    >>> cat(wheel_1.forward(l) for l in string.ascii_lowercase)
    'ekmflgdqvzntowyhxuspaibrcj'
    >>> cat(wheel_1.backward(l) for l in string.ascii_lowercase)
    'uwygadfpvzbeckmthxslrinqoj'


    >>> wheel_2 = SimpleWheel(wheel_ii_spec)
    >>> cat(wheel_2.forward(l) for l in string.ascii_lowercase)
    'ajdksiruxblhwtmcqgznpyfvoe'
    >>> cat(wheel_2.backward(l) for l in string.ascii_lowercase)
    'ajpczwrlfbdkotyuqgenhxmivs'

    >>> wheel_3 = SimpleWheel(wheel_iii_spec)
    >>> wheel_3.set_position('a')
    >>> wheel_3.advance()
    >>> cat(wheel_3.forward(l) for l in string.ascii_lowercase)
    'cegikboqswuymxdhvfzjltrpna'
    >>> cat(wheel_3.backward(l) for l in string.ascii_lowercase)
    'zfaobrcpdteumygxhwivkqjnls'
    >>> wheel_3.position
    1
    >>> wheel_3.position_l
    'b'

    >>> for _ in range(24): wheel_3.advance()
    >>> wheel_3.position
    25
    >>> wheel_3.position_l
    'z'
    >>> cat(wheel_3.forward(l) for l in string.ascii_lowercase)
    'pcegikmdqsuywaozfjxhblnvtr'
    >>> cat(wheel_3.backward(l) for l in string.ascii_lowercase)
    'nubhcqdterfvgwoaizjykxmslp'

    >>> wheel_3.advance()
    >>> wheel_3.position
    0
    >>> wheel_3.position_l
    'a'
    >>> cat(wheel_3.forward(l) for l in string.ascii_lowercase)
    'bdfhjlcprtxvznyeiwgakmusqo'
    >>> cat(wheel_3.backward(l) for l in string.ascii_lowercase)
    'tagbpcsdqeufvnzhyixjwlrkom'
    """
    def __init__(self, transform, position='a', raw_transform=False):
        super(SimpleWheel, self).__init__(transform, raw_transform)
        self.set_position(position)
        
    def __getattribute__(self,name):
        if name=='position_l':
            return unpos(self.position)
        else:
            return object.__getattribute__(self, name)
    
    def set_position(self, position):
        self.position = ord(position) - ord('a')
    
    def forward(self, letter):
        if letter in string.ascii_lowercase:
            return unpos((self.forward_map[(pos(letter) + self.position) % 26] - self.position))
        else:
            return ''
                
    def backward(self, letter):
        if letter in string.ascii_lowercase:
            return unpos((self.backward_map[(pos(letter) + self.position) % 26] - self.position))
        else:
            return ''
        
    def advance(self):
        self.position = (self.position + 1) % 26



class Wheel(SimpleWheel):
    """A wheel with a movable ring.

    The ring holds the letters and the pegs that turn other wheels. The core
    holds the wiring that does the transformation.

    The ring position is how many steps the core is turned relative to the ring.
    This is one-based, so a ring setting of 1 means the core and ring are 
    aligned.

    The position of the wheel is the position of the core (the transforms) 
    relative to the neutral position. 

    The position_l is the position of the ring, or what would be observed
    by the user of the Enigma machine. 

    The peg_positions are the number of advances of this wheel before it will 
    advance the next wheel.

    >>> wheel_3 = Wheel(wheel_iii_spec, wheel_iii_pegs, position='b', ring_setting=1)
    >>> wheel_3.position
    1
    >>> wheel_3.peg_positions
    [20]
    >>> wheel_3.position_l
    'b'
    >>> wheel_3.advance()
    >>> wheel_3.position
    2
    >>> wheel_3.peg_positions
    [19]
    >>> wheel_3.position_l
    'c'

    >>> wheel_6 = Wheel(wheel_vi_spec, wheel_vi_pegs, position='b', ring_setting=3)
    >>> cat(wheel_6.forward(l) for l in string.ascii_lowercase)
    'xkqhwpvngzrcfoiaselbtymjdu'
    >>> cat(wheel_6.backward(l) for l in string.ascii_lowercase)
    'ptlyrmidoxbswhnfckquzgeavj'
    >>> wheel_6.position
    25
    >>> 11 in wheel_6.peg_positions
    True
    >>> 24 in wheel_6.peg_positions
    True
    >>> wheel_6.position_l
    'b'

    >>> wheel_6.advance()
    >>> cat(wheel_6.forward(l) for l in string.ascii_lowercase)
    'jpgvoumfyqbenhzrdkasxlictw'
    >>> cat(wheel_6.backward(l) for l in string.ascii_lowercase)
    'skxqlhcnwarvgmebjptyfdzuio'
    >>> wheel_6.position
    0
    >>> 10 in wheel_6.peg_positions
    True
    >>> 23 in wheel_6.peg_positions
    True
    >>> wheel_6.position_l
    'c'

    >>> for _ in range(22): wheel_6.advance()
    >>> cat(wheel_6.forward(l) for l in string.ascii_lowercase)
    'mgxantkzsyqjcufirldvhoewbp'
    >>> cat(wheel_6.backward(l) for l in string.ascii_lowercase)
    'dymswobuplgraevzkqifntxcjh'
    >>> wheel_6.position
    22
    >>> 1 in wheel_6.peg_positions
    True
    >>> 14 in wheel_6.peg_positions
    True
    >>> wheel_6.position_l
    'y'

    >>> wheel_6.advance()
    >>> cat(wheel_6.forward(l) for l in string.ascii_lowercase)
    'fwzmsjyrxpibtehqkcugndvaol'
    >>> cat(wheel_6.backward(l) for l in string.ascii_lowercase)
    'xlrvnatokfqzduyjphemswbigc'
    >>> wheel_6.position
    23
    >>> 0 in wheel_6.peg_positions
    True
    >>> 13 in wheel_6.peg_positions
    True
    >>> wheel_6.position_l
    'z'

    >>> wheel_6.advance()
    >>> cat(wheel_6.forward(l) for l in string.ascii_lowercase)
    'vylrixqwohasdgpjbtfmcuznke'
    >>> cat(wheel_6.backward(l) for l in string.ascii_lowercase)
    'kqumzsnjepyctxiogdlrvahfbw'
    >>> wheel_6.position
    24
    >>> 25 in wheel_6.peg_positions
    True
    >>> 12 in wheel_6.peg_positions
    True
    >>> wheel_6.position_l
    'a'

    >>> wheel_6.advance()
    >>> cat(wheel_6.forward(l) for l in string.ascii_lowercase)
    'xkqhwpvngzrcfoiaselbtymjdu'
    >>> cat(wheel_6.backward(l) for l in string.ascii_lowercase)
    'ptlyrmidoxbswhnfckquzgeavj'
    >>> wheel_6.position
    25
    >>> 24 in wheel_6.peg_positions
    True
    >>> 11 in wheel_6.peg_positions
    True
    >>> wheel_6.position_l
    'b'

    >>> wheel_6.advance()
    >>> cat(wheel_6.forward(l) for l in string.ascii_lowercase)
    'jpgvoumfyqbenhzrdkasxlictw'
    >>> cat(wheel_6.backward(l) for l in string.ascii_lowercase)
    'skxqlhcnwarvgmebjptyfdzuio'
    >>> wheel_6.position
    0
    >>> 23 in wheel_6.peg_positions
    True
    >>> 10 in wheel_6.peg_positions
    True
    >>> wheel_6.position_l
    'c'

    """
    def __init__(self, transform, ring_peg_letters, ring_setting=1, position='a', raw_transform=False):
        self.ring_peg_letters = ring_peg_letters
        self.ring_setting = ring_setting
        super(Wheel, self).__init__(transform, position=position, raw_transform=raw_transform)
        self.set_position(position)
        
    def __getattribute__(self,name):
        if name=='position_l':
            return unpos(self.position + self.ring_setting - 1)
        else:
            return object.__getattribute__(self, name)

    def set_position(self, position):
        self.position = (pos(position) - self.ring_setting + 1) % 26
        self.peg_positions = [(pos(p) - pos(position)) % 26  for p in self.ring_peg_letters]
        
    def advance(self):
        super(Wheel, self).advance()
        self.peg_positions = [(p - 1) % 26 for p in self.peg_positions]


class Enigma(object):
    """An Enigma machine.

    >>> enigma = Enigma(reflector_b_spec, \
                wheel_i_spec, wheel_i_pegs, \
                wheel_ii_spec, wheel_ii_pegs, \
                wheel_iii_spec, wheel_iii_pegs, \
                1, 1, 1, \
                '')
    >>> enigma.set_wheels('a', 'a', 't')
    >>> enigma.wheel_positions
    (0, 0, 19)
    >>> cat(enigma.wheel_positions_l)
    'aat'
    >>> enigma.peg_positions
    ([16], [4], [2])
    >>> cat(enigma.lookup(l) for l in string.ascii_lowercase)
    'puvioztjdhxmlyeawsrgbcqknf'

    >>> enigma.advance()
    >>> enigma.wheel_positions
    (0, 0, 20)
    >>> cat(enigma.wheel_positions_l)
    'aau'
    >>> enigma.peg_positions
    ([16], [4], [1])
    >>> cat(enigma.lookup(l) for l in string.ascii_lowercase)
    'baigpldqcowfyzjehvtsxrkumn'

    >>> enigma.advance()
    >>> enigma.wheel_positions
    (0, 0, 21)
    >>> cat(enigma.wheel_positions_l)
    'aav'
    >>> enigma.peg_positions
    ([16], [4], [0])
    >>> cat(enigma.lookup(l) for l in string.ascii_lowercase)
    'mnvfydiwgzsoablrxpkutchqej'

    >>> enigma.advance()
    >>> enigma.wheel_positions
    (0, 1, 22)
    >>> cat(enigma.wheel_positions_l)
    'abw'
    >>> enigma.peg_positions
    ([16], [3], [25])
    >>> cat(enigma.lookup(l) for l in string.ascii_lowercase)
    'ulfopcykswhbzvderqixanjtgm'

    >>> enigma.advance()
    >>> enigma.wheel_positions
    (0, 1, 23)
    >>> cat(enigma.wheel_positions_l)
    'abx'
    >>> enigma.peg_positions
    ([16], [3], [24])
    >>> cat(enigma.lookup(l) for l in string.ascii_lowercase)
    'qmwftdyovursbzhxaklejicpgn'

    >>> enigma.advance()
    >>> enigma.wheel_positions
    (0, 1, 24)
    >>> cat(enigma.wheel_positions_l)
    'aby'
    >>> enigma.peg_positions
    ([16], [3], [23])
    >>> cat(enigma.lookup(l) for l in string.ascii_lowercase)
    'oljmzxrvucybdqasngpwihtfke'




    >>> enigma.set_wheels('a', 'd', 't')
    >>> enigma.wheel_positions
    (0, 3, 19)
    >>> cat(enigma.wheel_positions_l)
    'adt'
    >>> enigma.peg_positions
    ([16], [1], [2])
    >>> cat(enigma.lookup(l) for l in string.ascii_lowercase)
    'zcbpqxwsjiuonmldethrkygfva'

    >>> enigma.advance()
    >>> enigma.wheel_positions
    (0, 3, 20)
    >>> cat(enigma.wheel_positions_l)
    'adu'
    >>> enigma.peg_positions
    ([16], [1], [1])
    >>> cat(enigma.lookup(l) for l in string.ascii_lowercase)
    'ehprawjbngotxikcsdqlzyfmvu'

    >>> enigma.advance()
    >>> enigma.wheel_positions
    (0, 3, 21)
    >>> cat(enigma.wheel_positions_l)
    'adv'
    >>> enigma.peg_positions
    ([16], [1], [0])
    >>> cat(enigma.lookup(l) for l in string.ascii_lowercase)
    'eqzxarpihmnvjkwgbfuyslodtc'

    >>> enigma.advance()
    >>> enigma.wheel_positions
    (0, 4, 22)
    >>> cat(enigma.wheel_positions_l)
    'aew'
    >>> enigma.peg_positions
    ([16], [0], [25])
    >>> cat(enigma.lookup(l) for l in string.ascii_lowercase)
    'qedcbtpluzmhkongavwfirsyxj'

    >>> enigma.advance()
    >>> enigma.wheel_positions
    (1, 5, 23)
    >>> cat(enigma.wheel_positions_l)
    'bfx'
    >>> enigma.peg_positions
    ([15], [25], [24])
    >>> cat(enigma.lookup(l) for l in string.ascii_lowercase)
    'iwuedhsfazqxytvrkpgncoblmj'

    >>> enigma.advance()
    >>> enigma.wheel_positions
    (1, 5, 24)
    >>> cat(enigma.wheel_positions_l)
    'bfy'
    >>> enigma.peg_positions
    ([15], [25], [23])
    >>> cat(enigma.lookup(l) for l in string.ascii_lowercase)
    'baknstqzrmcxjdvygiefwoulph'


    >>> enigma.set_wheels('a', 'a', 'a')
    >>> ct = enigma.encipher('testmessage')
    >>> ct
    'olpfhnvflyn'

    >>> enigma.set_wheels('a', 'd', 't')
    >>> ct = enigma.encipher('testmessage')
    >>> ct
    'lawnjgpwjik'


    >>> enigma.set_wheels('b', 'd', 'q')
    >>> ct = enigma.encipher('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
    >>> ct
    'kvmmwrlqlqsqpeugjrcxzwpfyiyybwloewrouvkpoztceuwtfjzqwpbqldttsr'
    >>> enigma.left_wheel.position_l
    'c'
    >>> enigma.middle_wheel.position_l
    'h'
    >>> enigma.right_wheel.position_l
    'a'

    # Setting sheet line 31 from http://www.codesandciphers.org.uk/enigma/enigma3.htm
    # Enigma simulation settings are 
    # http://enigma.louisedade.co.uk/enigma.html?m3;b;b153;AFTX;AJEU;AU-BG-EY-FP-HL-IN-JZ-OS-QR-TX
    >>> enigma31 = Enigma(reflector_b_spec, \
                wheel_i_spec, wheel_i_pegs, \
                wheel_v_spec, wheel_v_pegs, \
                wheel_iii_spec, wheel_iii_pegs, \
                6, 20, 24, \
                'ua pf rq so ni ey bg hl tx zj')

    >>> enigma31.set_wheels('j', 'e', 'u')

    >>> enigma31.advance()
    >>> enigma31.wheel_positions
    (4, 11, 24)
    >>> cat(enigma31.wheel_positions_l)
    'jev'
    >>> enigma31.peg_positions
    ([7], [21], [0])
    >>> cat(enigma31.lookup(l) for l in string.ascii_lowercase)
    'mvqjlyowkdieasgzcunxrbhtfp'

    >>> enigma31.advance()
    >>> enigma31.wheel_positions
    (4, 12, 25)
    >>> cat(enigma31.wheel_positions_l)
    'jfw'
    >>> enigma31.peg_positions
    ([7], [20], [25])
    >>> cat(enigma31.lookup(l) for l in string.ascii_lowercase)
    'sjolzuyvrbwdpxcmtiaqfhknge'

    >>> enigma31.advance()
    >>> enigma31.wheel_positions
    (4, 12, 0)
    >>> cat(enigma31.wheel_positions_l)
    'jfx'
    >>> enigma31.peg_positions
    ([7], [20], [24])
    >>> cat(enigma31.lookup(l) for l in string.ascii_lowercase)
    'qrxedkoywufmlvgsabpzjnicht'

    >>> enigma31.advance()
    >>> enigma31.wheel_positions
    (4, 12, 1)
    >>> cat(enigma31.wheel_positions_l)
    'jfy'
    >>> enigma31.peg_positions
    ([7], [20], [23])
    >>> cat(enigma31.lookup(l) for l in string.ascii_lowercase)
    'hpsukliagqefwvtbjxcodnmrzy'

    >>> enigma31.advance()
    >>> enigma31.wheel_positions
    (4, 12, 2)
    >>> cat(enigma31.wheel_positions_l)
    'jfz'
    >>> enigma31.peg_positions
    ([7], [20], [22])
    >>> cat(enigma31.lookup(l) for l in string.ascii_lowercase)
    'zevnbpyqowrtxdifhkulscjmga'


    >>> enigma31.set_wheels('i', 'd', 'z')

    >>> enigma31.advance()
    >>> enigma31.wheel_positions
    (3, 10, 3)
    >>> cat(enigma31.wheel_positions_l)
    'ida'
    >>> enigma31.peg_positions
    ([8], [22], [21])
    >>> cat(enigma31.lookup(l) for l in string.ascii_lowercase)
    'ikhpqrvcambzjondefwyxgsutl'

    >>> enigma31.advance()
    >>> enigma31.wheel_positions
    (3, 10, 4)
    >>> cat(enigma31.wheel_positions_l)
    'idb'
    >>> enigma31.peg_positions
    ([8], [22], [20])
    >>> cat(enigma31.lookup(l) for l in string.ascii_lowercase)
    'cdabskhgzwfmlqvunyexpojtri'

    >>> enigma31.advance()
    >>> enigma31.wheel_positions
    (3, 10, 5)
    >>> cat(enigma31.wheel_positions_l)
    'idc'
    >>> enigma31.peg_positions
    ([8], [22], [19])
    >>> cat(enigma31.lookup(l) for l in string.ascii_lowercase)
    'pcbwiqhgemyvjsuaftnroldzkx'

    >>> enigma31.advance()
    >>> enigma31.wheel_positions
    (3, 10, 6)
    >>> cat(enigma31.wheel_positions_l)
    'idd'
    >>> enigma31.peg_positions
    ([8], [22], [18])
    >>> cat(enigma31.lookup(l) for l in string.ascii_lowercase)
    'xcbfvdnouptmlghjzwykierasq'

    >>> enigma31.advance()
    >>> enigma31.wheel_positions
    (3, 10, 7)
    >>> cat(enigma31.wheel_positions_l)
    'ide'
    >>> enigma31.peg_positions
    ([8], [22], [17])
    >>> cat(enigma31.lookup(l) for l in string.ascii_lowercase)
    'xfvglbdynuseriwqpmkzjcoaht'

    >>> enigma31.advance()
    >>> enigma31.wheel_positions
    (3, 10, 8)
    >>> cat(enigma31.wheel_positions_l)
    'idf'
    >>> enigma31.peg_positions
    ([8], [22], [16])
    >>> cat(enigma31.lookup(l) for l in string.ascii_lowercase)
    'tfpqlbouynsewjgcdxkahzmriv'

    >>> enigma31.advance()
    >>> enigma31.wheel_positions
    (3, 10, 9)
    >>> cat(enigma31.wheel_positions_l)
    'idg'
    >>> enigma31.peg_positions
    ([8], [22], [15])
    >>> cat(enigma31.lookup(l) for l in string.ascii_lowercase)
    'cjaunvlwtbygzexrspqidfhokm'

    >>> enigma31.advance()
    >>> enigma31.wheel_positions
    (3, 10, 10)
    >>> cat(enigma31.wheel_positions_l)
    'idh'
    >>> enigma31.peg_positions
    ([8], [22], [14])
    >>> cat(enigma31.lookup(l) for l in string.ascii_lowercase)
    'yltxkrqvowebzpingfucshjdam'

    >>> enigma31.advance()
    >>> enigma31.wheel_positions
    (3, 10, 11)
    >>> cat(enigma31.wheel_positions_l)
    'idi'
    >>> enigma31.peg_positions
    ([8], [22], [13])
    >>> cat(enigma31.lookup(l) for l in string.ascii_lowercase)
    'myktluzrnxceaiqsohpdfwvjbg'

    >>> enigma31.advance()
    >>> enigma31.wheel_positions
    (3, 10, 12)
    >>> cat(enigma31.wheel_positions_l)
    'idj'
    >>> enigma31.peg_positions
    ([8], [22], [12])
    >>> cat(enigma31.lookup(l) for l in string.ascii_lowercase)
    'pynjrmiugdqxfcvakewzhoslbt'

    >>> enigma31.advance()
    >>> enigma31.wheel_positions
    (3, 10, 13)
    >>> cat(enigma31.wheel_positions_l)
    'idk'
    >>> enigma31.peg_positions
    ([8], [22], [11])
    >>> cat(enigma31.lookup(l) for l in string.ascii_lowercase)
    'mwvedyplnoxhaijgrqtszcbkfu'

    >>> enigma31.advance()
    >>> enigma31.wheel_positions
    (3, 10, 14)
    >>> cat(enigma31.wheel_positions_l)
    'idl'
    >>> enigma31.peg_positions
    ([8], [22], [10])
    >>> cat(enigma31.lookup(l) for l in string.ascii_lowercase)
    'qcbrfeutvoxpnmjladzhgiykws'

    >>> enigma31.advance()
    >>> enigma31.wheel_positions
    (3, 10, 15)
    >>> cat(enigma31.wheel_positions_l)
    'idm'
    >>> enigma31.peg_positions
    ([8], [22], [9])
    >>> cat(enigma31.lookup(l) for l in string.ascii_lowercase)
    'dnoahryetsmukbcvwfjilpqzgx'

    >>> enigma31.advance()
    >>> enigma31.wheel_positions
    (3, 10, 16)
    >>> cat(enigma31.wheel_positions_l)
    'idn'
    >>> enigma31.peg_positions
    ([8], [22], [8])
    >>> cat(enigma31.lookup(l) for l in string.ascii_lowercase)
    'nidcfehgbqsovalyjzkxwmutpr'

    >>> enigma31.advance()
    >>> enigma31.wheel_positions
    (3, 10, 17)
    >>> cat(enigma31.wheel_positions_l)
    'ido'
    >>> enigma31.peg_positions
    ([8], [22], [7])
    >>> cat(enigma31.lookup(l) for l in string.ascii_lowercase)
    'joifxdulcarhzpbntkwqgysevm'

    >>> enigma31.advance()
    >>> enigma31.wheel_positions
    (3, 10, 18)
    >>> cat(enigma31.wheel_positions_l)
    'idp'
    >>> enigma31.peg_positions
    ([8], [22], [6])
    >>> cat(enigma31.lookup(l) for l in string.ascii_lowercase)
    'ptnlsxvozmwdjchayuebrgkfqi'

    >>> enigma31.advance()
    >>> enigma31.wheel_positions
    (3, 10, 19)
    >>> cat(enigma31.wheel_positions_l)
    'idq'
    >>> enigma31.peg_positions
    ([8], [22], [5])
    >>> cat(enigma31.lookup(l) for l in string.ascii_lowercase)
    'slwopzqnmxybihdeguavrtcjkf'

    >>> enigma31.advance()
    >>> enigma31.wheel_positions
    (3, 10, 20)
    >>> cat(enigma31.wheel_positions_l)
    'idr'
    >>> enigma31.peg_positions
    ([8], [22], [4])
    >>> cat(enigma31.lookup(l) for l in string.ascii_lowercase)
    'hcbedwlamzogixkytsrqvufnpj'

    >>> enigma31.advance()
    >>> enigma31.wheel_positions
    (3, 10, 21)
    >>> cat(enigma31.wheel_positions_l)
    'ids'
    >>> enigma31.peg_positions
    ([8], [22], [3])
    >>> cat(enigma31.lookup(l) for l in string.ascii_lowercase)
    'odxbjwzrmelkisavuhnyqpfctg'

    >>> enigma31.advance()
    >>> enigma31.wheel_positions
    (3, 10, 22)
    >>> cat(enigma31.wheel_positions_l)
    'idt'
    >>> enigma31.peg_positions
    ([8], [22], [2])
    >>> cat(enigma31.lookup(l) for l in string.ascii_lowercase)
    'udgbfeclrwnhxksvtioqapjmzy'

    >>> enigma31.advance()
    >>> enigma31.wheel_positions
    (3, 10, 23)
    >>> cat(enigma31.wheel_positions_l)
    'idu'
    >>> enigma31.peg_positions
    ([8], [22], [1])
    >>> cat(enigma31.lookup(l) for l in string.ascii_lowercase)
    'nrdczqxmowvshaiufblypkjgte'

    >>> enigma31.advance()
    >>> enigma31.wheel_positions
    (3, 10, 24)
    >>> cat(enigma31.wheel_positions_l)
    'idv'
    >>> enigma31.peg_positions
    ([8], [22], [0])
    >>> cat(enigma31.lookup(l) for l in string.ascii_lowercase)
    'hkifjdoacebqtzgulyvmpsxwrn'

    >>> enigma31.advance()
    >>> enigma31.wheel_positions
    (3, 11, 25)
    >>> cat(enigma31.wheel_positions_l)
    'iew'
    >>> enigma31.peg_positions
    ([8], [21], [25])
    >>> cat(enigma31.lookup(l) for l in string.ascii_lowercase)
    'yptzuhofqvnmlkgbixwcejsrad'

    >>> enigma31.advance()
    >>> enigma31.wheel_positions
    (3, 11, 0)
    >>> cat(enigma31.wheel_positions_l)
    'iex'
    >>> enigma31.peg_positions
    ([8], [21], [24])
    >>> cat(enigma31.lookup(l) for l in string.ascii_lowercase)
    'vkdcwhqfjibzsptngumoraeyxl'

    >>> enigma31.advance()
    >>> enigma31.wheel_positions
    (3, 11, 1)
    >>> cat(enigma31.wheel_positions_l)
    'iey'
    >>> enigma31.peg_positions
    ([8], [21], [23])
    >>> cat(enigma31.lookup(l) for l in string.ascii_lowercase)
    'wenpbqrouxlkychdfgzvitajms'


    >>> enigma31.set_wheels('i', 'z', 'd')
    >>> enigma31.encipher('verylongtestmessagewithanextrabitofmessageforgoodmeasure')
    'apocwtjuikurcfivlozvhffkoacxufcekthcvodfqpxdjqyckdozlqki'
    >>> enigma31.wheel_positions
    (4, 9, 10)
    >>> cat(enigma31.wheel_positions_l)
    'jch'
    >>> enigma31.peg_positions
    ([7], [23], [14])
    >>> cat(enigma31.lookup(l) for l in string.ascii_lowercase)
    'mopnigfuesqwadbcktjrhylzvx'

    >>> enigma31.set_wheels('i', 'z', 'd')
    >>> enigma31.decipher('apocwtjuikurcfivlozvhffkoacxufcekthcvodfqpxdjqyckdozlqki')
    'verylongtestmessagewithanextrabitofmessageforgoodmeasure'
    """
    def __init__(self, reflector_spec,
                 left_wheel_spec, left_wheel_pegs,
                 middle_wheel_spec, middle_wheel_pegs,
                 right_wheel_spec, right_wheel_pegs,
                 left_ring_setting, middle_ring_setting, right_ring_setting,
                 plugboard_setting):
        self.reflector = Reflector(reflector_spec)
        self.left_wheel = Wheel(left_wheel_spec, left_wheel_pegs, ring_setting=left_ring_setting)
        self.middle_wheel = Wheel(middle_wheel_spec, middle_wheel_pegs, ring_setting=middle_ring_setting)
        self.right_wheel = Wheel(right_wheel_spec, right_wheel_pegs, ring_setting=right_ring_setting)
        self.plugboard = Plugboard(plugboard_setting)
        
    def __getattribute__(self,name):
        if name=='wheel_positions':
            return self.left_wheel.position, self.middle_wheel.position, self.right_wheel.position 
        elif name=='wheel_positions_l':
            return self.left_wheel.position_l, self.middle_wheel.position_l, self.right_wheel.position_l 
        elif name=='peg_positions':
            return self.left_wheel.peg_positions, self.middle_wheel.peg_positions, self.right_wheel.peg_positions
        else:
            return object.__getattribute__(self, name)

    def set_wheels(self, left_wheel_position, middle_wheel_position, right_wheel_position):
        self.left_wheel.set_position(left_wheel_position)
        self.middle_wheel.set_position(middle_wheel_position)
        self.right_wheel.set_position(right_wheel_position)
        
    def lookup(self, letter):
        a = self.plugboard.forward(letter)
        b = self.right_wheel.forward(a)
        c = self.middle_wheel.forward(b)
        d = self.left_wheel.forward(c)
        e = self.reflector.forward(d)
        f = self.left_wheel.backward(e)
        g = self.middle_wheel.backward(f)
        h = self.right_wheel.backward(g)
        i = self.plugboard.backward(h)
        return i
    
    def advance(self):
        advance_middle = False
        advance_left = False
        if 0 in self.right_wheel.peg_positions:
            advance_middle = True
        if 0 in self.middle_wheel.peg_positions:
            advance_left = True
            advance_middle = True
        self.right_wheel.advance()
        if advance_middle: self.middle_wheel.advance()
        if advance_left: self.left_wheel.advance()
            
    def encipher_letter(self, letter):
        self.advance()
        return self.lookup(letter)
    
    def encipher(self, message):
        enciphered = ''
        for letter in clean(message):
            enciphered += self.encipher_letter(letter)
        return enciphered

    decipher = encipher


# for i in range(26):
#     enigma.advance()
#     print('enigma.advance()')
#     print("assert(enigma.wheel_positions == {})".format(enigma.wheel_positions))
#     print("assert(cat(enigma.wheel_positions_l) == '{}')".format(cat(enigma.wheel_positions_l)))
#     print("assert(enigma.peg_positions == {})".format(enigma.peg_positions))
#     print("assert(cat(enigma.lookup(l) for l in string.ascii_lowercase) == '{}')".format(cat(enigma.lookup(l) for l in string.ascii_lowercase)))
#     print()


##################################
# # Bombe
##################################

Signal = collections.namedtuple('Signal', ['bank', 'wire'])
Connection = collections.namedtuple('Connection', ['banks', 'scrambler'])
MenuItem = collections.namedtuple('MenuIem', ['before', 'after', 'number'])


class Scrambler(object):
    def __init__(self, wheel1_spec, wheel2_spec, wheel3_spec, reflector_spec,
                 wheel1_pos='a', wheel2_pos='a', wheel3_pos='a'):
        self.wheel1 = SimpleWheel(wheel1_spec, position=wheel1_pos)
        self.wheel2 = SimpleWheel(wheel2_spec, position=wheel2_pos)
        self.wheel3 = SimpleWheel(wheel3_spec, position=wheel3_pos)
        self.reflector = Reflector(reflector_spec)
    
    def __getattribute__(self, name):
        if name=='wheel_positions':
            return self.wheel1.position, self.wheel2.position, self.wheel3.position 
        elif name=='wheel_positions_l':
            return self.wheel1.position_l, self.wheel2.position_l, self.wheel3.position_l 
        else:
            return object.__getattribute__(self, name)
    
    def advance(self, wheel1=False, wheel2=False, wheel3=True):
        if wheel1: self.wheel1.advance()
        if wheel2: self.wheel2.advance()
        if wheel3: self.wheel3.advance()
            
    def lookup(self, letter):
        a = self.wheel3.forward(letter)
        b = self.wheel2.forward(a)
        c = self.wheel1.forward(b)
        d = self.reflector.forward(c)
        e = self.wheel1.backward(d)
        f = self.wheel2.backward(e)
        g = self.wheel3.backward(f)
        return g
    
    def set_positions(self, wheel1_pos, wheel2_pos, wheel3_pos):
        self.wheel1.set_position(wheel1_pos)
        self.wheel2.set_position(wheel2_pos)
        self.wheel3.set_position(wheel3_pos)      


class Bombe(object):
    def __init__(self, wheel1_spec, wheel2_spec, wheel3_spec, reflector_spec,
                menu=None, start_signal=None, use_diagonal_board=True, 
                verify_plugboard=True):
        self.connections = []
        self.wheel1_spec = wheel1_spec
        self.wheel2_spec = wheel2_spec
        self.wheel3_spec = wheel3_spec
        self.reflector_spec = reflector_spec
        if menu:
            self.read_menu(menu)
        if start_signal:
            self.test_start = start_signal
        self.use_diagonal_board = use_diagonal_board
        self.verify_plugboard = verify_plugboard
        
    def __getattribute__(self, name):
        if name=='wheel_positions':
            return self.connections[0].scrambler.wheel_positions
        elif name=='wheel_positions_l':
            return self.connections[0].scrambler.wheel_positions_l
        else:
            return object.__getattribute__(self, name)
        
    def __call__(self, start_positions):
        return start_positions, self.test(initial_signal=self.test_start,
            start_positions=start_positions, 
            use_diagonal_board=self.use_diagonal_board,
            verify_plugboard=self.verify_plugboard)
        
    def add_connection(self, bank_before, bank_after, scrambler):
        self.connections += [Connection([bank_before, bank_after], scrambler)]
        
    def read_menu(self, menu):
        for item in menu:
            scrambler = Scrambler(self.wheel1_spec, self.wheel2_spec, self.wheel3_spec,
                                  self.reflector_spec,
                                  wheel3_pos=unpos(item.number - 1))
            self.add_connection(item.before, item.after, scrambler)
        most_common_letter = (collections.Counter(m.before for m in menu) + \
                              collections.Counter(m.after for m in menu)).most_common(1)[0][0]
        self.test_start = Signal(most_common_letter, most_common_letter)
        
    def set_positions(self, wheel1_pos, wheel2_pos, wheel3_pos):
        for i, c in enumerate(self.connections):
            c.scrambler.set_positions(wheel1_pos, wheel2_pos, unpos(pos(wheel3_pos) + i))
    
    def test(self, initial_signal=None, start_positions=None, use_diagonal_board=True,
            verify_plugboard=True):
        self.banks = {label: 
                      dict(zip(string.ascii_lowercase, [False]*len(string.ascii_lowercase)))
                      for label in string.ascii_lowercase}
        if start_positions:
            self.set_positions(*start_positions)
        if not initial_signal:
            initial_signal = self.test_start
        self.pending = [initial_signal]
        self.propagate(use_diagonal_board)
        live_wire_count = len([self.banks[self.test_start.bank][w] 
                    for w in self.banks[self.test_start.bank] 
                    if self.banks[self.test_start.bank][w]])
        if live_wire_count < 26:
            if verify_plugboard:
                possibles = self.possible_plugboards()
                return all(s0.isdisjoint(s1) for s0 in possibles for s1 in possibles if s0 != s1)
            else:
                return True
        else:
            return False
        
    def propagate(self, use_diagonal_board):
        while self.pending:
            current = self.pending[0]
            # print("processing", current)
            self.pending = self.pending[1:]
            if not self.banks[current.bank][current.wire]:
                self.banks[current.bank][current.wire] = True
                if use_diagonal_board:
                    self.pending += [Signal(current.wire, current.bank)]
                for c in self.connections:
                    if current.bank in c.banks:
                        other_bank = [b for b in c.banks if b != current.bank][0]
                        other_wire = c.scrambler.lookup(current.wire)
                        # print("  adding", other_bank, other_wire, "because", c.banks)
                        self.pending += [Signal(other_bank, other_wire)]
    
    def run(self, run_start=None, wheel1_pos='a', wheel2_pos='a', wheel3_pos='a', use_diagonal_board=True):
        if not run_start:
            run_start = self.test_start
        self.solutions = []
        self.set_positions(wheel1_pos, wheel2_pos, wheel3_pos)
        for run_index in range(26*26*26):
            if self.test(initial_signal=run_start, use_diagonal_board=use_diagonal_board):
                self.solutions += [self.connections[0].scrambler.wheel_positions_l]
            advance3 = True
            advance2 = False
            advance1 = False
            if (run_index + 1) % 26 == 0: advance2 = True
            if (run_index + 1) % (26*26) == 0: advance1 = True
            for c in self.connections:
                c.scrambler.advance(advance1, advance2, advance3)
        return self.solutions
    
    def possible_plugboards(self):
        possibles = set()
        for b in self.banks:
            active = [w for w in self.banks[b] if self.banks[b][w]]
            inactive = [w for w in self.banks[b] if not self.banks[b][w]]
            if len(active) == 1:
                possibles = possibles.union({frozenset((b, active[0]))})
            if len(inactive) == 1:
                possibles = possibles.union({frozenset((b, inactive[0]))})
        return possibles


def make_menu(plaintext, ciphertext):
    return [MenuItem(p, c, i+1) 
            for i, (p, c) in enumerate(zip(plaintext, ciphertext))]


def run_multi_bombe(wheel1_spec, wheel2_spec, wheel3_spec, reflector_spec, menu,
                    start_signal=None, use_diagonal_board=True, 
                    verify_plugboard=True):
    allwheels = itertools.product(string.ascii_lowercase, repeat=3)

    with multiprocessing.Pool() as pool:
        res = pool.map(Bombe(wheel1_spec, wheel2_spec, wheel3_spec, 
            reflector_spec, menu=menu, start_signal=start_signal, 
            use_diagonal_board=use_diagonal_board, 
            verify_plugboard=verify_plugboard),
                  allwheels)
    return [r[0] for r in res if r[1]]


if __name__ == "__main__":
    import doctest
    # doctest.testmod(extraglobs={'lt': LetterTransformer(1, 'a')})
    doctest.testmod()


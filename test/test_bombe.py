import unittest
import string 

from cipher.enigma import *
from cipher.bombe import *

class ScramblerTest(unittest.TestCase):
    def setUp(self):
        self.scrambler = Scrambler(wheel_i_spec, wheel_ii_spec, 
            wheel_iii_spec, reflector_b_spec)

    def test_attributes(self):
        self.assertEqual(self.scrambler.wheel_positions, (0, 0, 0))
        self.assertEqual(self.scrambler.wheel_positions_l, ('a', 'a', 'a'))

    def test_set_positions(self):
        self.scrambler.set_positions(1, 2, 3)
        self.assertEqual(self.scrambler.wheel_positions, (1, 2, 3))
        self.assertEqual(self.scrambler.wheel_positions_l, ('b', 'c', 'd'))
        self.scrambler.set_positions('p', 'q', 'r')
        self.assertEqual(self.scrambler.wheel_positions, (15, 16, 17))
        self.assertEqual(self.scrambler.wheel_positions_l, ('p', 'q', 'r'))

    def test_advance(self):
        self.assertEqual(self.scrambler.wheel_positions, (0, 0, 0))
        self.scrambler.advance()
        self.assertEqual(self.scrambler.wheel_positions, (0, 0, 1))
        self.scrambler.advance()
        self.assertEqual(self.scrambler.wheel_positions, (0, 0, 2))
        self.scrambler.set_positions(0, 0, 25)
        self.assertEqual(self.scrambler.wheel_positions, (0, 0, 25))
        self.scrambler.advance()
        self.assertEqual(self.scrambler.wheel_positions, (0, 0, 0))
        self.scrambler.set_positions(0, 0, 25)
        self.scrambler.advance(wheel3=False)
        self.assertEqual(self.scrambler.wheel_positions, (0, 0, 25))
        self.scrambler.set_positions(0, 0, 25)
        self.scrambler.advance(wheel2=True)
        self.assertEqual(self.scrambler.wheel_positions, (0, 1, 0))
        self.scrambler.set_positions(0, 0, 25)
        self.scrambler.advance(wheel1=True, wheel2=True)
        self.assertEqual(self.scrambler.wheel_positions, (1, 1, 0))

    def test_lookups(self):
        self.scrambler.set_positions(0, 0, 0)
        self.assertEqual(cat(self.scrambler.lookup(l) 
                for l in string.ascii_lowercase),
            'uejobtpzwcnsrkdgvmlfaqiyxh')
        self.assertEqual(cat(self.scrambler.lookup(l) 
                for l in 'uejobtpzwcnsrkdgvmlfaqiyxh'),
            'abcdefghijklmnopqrstuvwxyz')
        self.scrambler.set_positions('p', 'q', 'r')
        self.assertEqual(cat(self.scrambler.lookup(l) 
                for l in string.ascii_lowercase),
            'jgqmnwbtvaurdezxclyhkifpso')
        self.assertEqual(cat(self.scrambler.lookup(l) 
                for l in 'jgqmnwbtvaurdezxclyhkifpso'),
            'abcdefghijklmnopqrstuvwxyz')
    
class BombeTest(unittest.TestCase):
    def setUp(self):
        self.bombe = Bombe(wheel_i_spec, wheel_ii_spec, 
            wheel_iii_spec, reflector_b_spec)
        self.plaintext = 'thisisatestmessage'
        self.ciphertext = 'opgndxcrwomnlnecjz'
        self.menu = make_menu(self.plaintext, self.ciphertext)
        self.bombe.read_menu(self.menu)

    def test_menu(self):
        self.assertEqual(len(self.bombe.connections), 18)
        self.assertEqual(':'.join(sorted(cat(sorted(c.banks))
                for c in self.bombe.connections)),
            'ac:ac:di:el:es:ew:ez:gi:gj:hp:mn:mt:ns:ns:os:ot:rt:sx')
        self.assertEqual(':'.join(sorted(cat(c.scrambler.wheel_positions_l)
                for c in self.bombe.connections)),
            'aaa:aab:aac:aad:aae:aaf:aag:aah:aai:aaj:aak:aal:aam:aan:aao:aap:aaq:aar')

        self.bombe.read_menu(self.menu)
        self.assertEqual(len(self.bombe.connections), 18)

    def test_signal(self):
        self.bombe.test(Signal('t', 't'))
        self.assertEqual(len(self.bombe.banks['t']), 26)
        self.assertTrue(all(self.bombe.banks['t'].values()))
        self.assertEqual(sum(1 for s in self.bombe.banks['u'].values() if s), 18)

        self.bombe.set_positions('a', 'a', 'b')
        self.bombe.test()
        self.assertEqual(sum(1 for b in self.bombe.banks 
                for s in self.bombe.banks[b].values() if s),
            11)

    def test_valid_with_rings(self):
        pt31 = 'someplaintext'
        ct31 = 'dhnpforeeimgg'
        menu31 = make_menu(pt31, ct31)
        b31 = Bombe(wheel_i_spec, wheel_v_spec, wheel_iii_spec, reflector_b_spec)
        b31.read_menu(menu31)
        b31.set_positions('e', 'l', 'f')

        b31.test(Signal('s', 'o'))
        self.assertEqual(sum(1 for b in b31.banks 
                for s in b31.banks[b].values() if s),
            5)
        self.assertEqual(':'.join(sorted(cat(sorted(p)) 
                for p in b31.possible_plugboards())),
            'd:hl:os')

        b31.test(Signal('o', 'o'))
        self.assertEqual(sum(1 for b in b31.banks 
                for s in b31.banks[b].values() if s),
            507)
        self.assertEqual(':'.join(sorted(cat(sorted(p)) 
                for p in b31.possible_plugboards())),
            'bg:ey:fp:in:m:tx')

    def test_invalid_with_rings(self):
        pt31 = 'someplaintext'
        ct31 = 'dhnpforeeimgg'
        menu31 = make_menu(pt31, ct31)
        b31 = Bombe(wheel_i_spec, wheel_v_spec, wheel_iii_spec, reflector_b_spec)
        b31.read_menu(menu31)
        b31.set_positions('a', 'a', 'a')

        b31.test(Signal('a', 'o'))
        self.assertEqual(sum(1 for b in b31.banks 
                for s in b31.banks[b].values() if s),
            514)
        self.assertEqual(':'.join(sorted(cat(sorted(p)) 
                for p in b31.possible_plugboards())),
            '')

if __name__ == '__main__':
    unittest.main()

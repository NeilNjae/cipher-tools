import unittest
import string 

from cipher.affine import *
from support.utilities import *

class AffineTest(unittest.TestCase):

    def test_encipher_letter(self):
        for p, c in zip(
                string.ascii_letters, 
                'hknqtwzcfiloruxadgjmpsvybeHKNQTWZCFILORUXADGJMPSVYBE'):
            self.assertEqual(affine_encipher_letter(p, 3, 5, True), c)

        for p, c in zip(
                string.ascii_letters, 
                'filoruxadgjmpsvybehknqtwzcFILORUXADGJMPSVYBEHKNQTWZC'):
            self.assertEqual(affine_encipher_letter(p, 3, 5, False), c)


    def test_decipher_letter(self):
        for p, c in zip(
                string.ascii_letters, 
                'hknqtwzcfiloruxadgjmpsvybeHKNQTWZCFILORUXADGJMPSVYBE'):
            self.assertEqual(affine_decipher_letter(c, 3, 5, True), p)

        for p, c in zip(
                string.ascii_letters, 
                'filoruxadgjmpsvybehknqtwzcFILORUXADGJMPSVYBEHKNQTWZC'):
            self.assertEqual(affine_decipher_letter(c, 3, 5, False), p)

    def test_encipher_message(self):
        self.assertEqual(affine_encipher(
                'hours passed during which jerico tried every trick he could think of', 
                15, 22, True),
            'lmyfu bkuusd dyfaxw claol psfaom jfasd snsfg jfaoe ls omytd jlaxe mh')


    def test_decipher_message(self):
        self.assertEqual(affine_decipher('lmyfu bkuusd dyfaxw claol psfaom jfasd snsfg jfaoe ls omytd jlaxe mh', 
                    15, 22, True),
            'hours passed during which jerico tried every trick he could think of')


    def test_break(self):
        ciphertext = '''lmyfu bkuusd dyfaxw claol psfaom jfasd snsfg jfaoe ls 
          omytd jlaxe mh jm bfmibj umis hfsul axubafkjamx. ls kffkxwsd jls 
          ofgbjmwfkiu olfmxmtmwaokttg jlsx ls kffkxwsd jlsi zg tsxwjl. jlsx 
          ls umfjsd jlsi zg hfsqysxog. ls dmmdtsd mx jls bats mh bkbsf. ls 
          bfmctsd kfmyxd jls lyj, mztanamyu xmc jm clm cku tmmeaxw kj lai 
          kxd clm ckuxj.'''
        expected_key = (15, 22, True)
        expected_score = -340.6011819
        actual_key, actual_score = affine_break(ciphertext)
        self.assertEqual(expected_key, actual_key)
        self.assertAlmostEqual(expected_score, actual_score)

if __name__ == '__main__':
    unittest.main()

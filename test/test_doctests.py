import unittest
import doctest

import cipher.caesar
import cipher.affine
import cipher.keyword_cipher
import cipher.polybius
import cipher.column_transposition
import cipher.railfence
import cipher.cadenus
import cipher.hill
import cipher.amsco
import cipher.bifid
import cipher.autokey
import cipher.pocket_enigma


def load_tests(loader, tests, ignore):

    tests.addTests(doctest.DocTestSuite(cipher.caesar))
    tests.addTests(doctest.DocTestSuite(cipher.affine))
    tests.addTests(doctest.DocTestSuite(cipher.keyword_cipher))
    tests.addTests(doctest.DocTestSuite(cipher.polybius))
    tests.addTests(doctest.DocTestSuite(cipher.column_transposition))
    tests.addTests(doctest.DocTestSuite(cipher.railfence))
    tests.addTests(doctest.DocTestSuite(cipher.cadenus))
    tests.addTests(doctest.DocTestSuite(cipher.hill))
    tests.addTests(doctest.DocTestSuite(cipher.amsco))
    tests.addTests(doctest.DocTestSuite(cipher.bifid))
    tests.addTests(doctest.DocTestSuite(cipher.autokey))
    tests.addTests(doctest.DocTestSuite(cipher.pocket_enigma, 
        extraglobs={'pe': cipher.pocket_enigma.PocketEnigma(1, 'a')}))
    return tests

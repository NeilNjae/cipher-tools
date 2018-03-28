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

import support.language_models
import support.norms
import support.segment
import support.text_prettify
import support.utilities


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

    tests.addTests(doctest.DocTestSuite(support.language_models))
    tests.addTests(doctest.DocTestSuite(support.norms))
    tests.addTests(doctest.DocTestSuite(support.segment))
    tests.addTests(doctest.DocTestSuite(support.text_prettify))
    tests.addTests(doctest.DocTestSuite(support.utilities))

    return tests

import unittest
import doctest
import cipher

def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(cipher))
    return tests

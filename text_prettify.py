from segment import segment
from cipher import cat
from language_models import sanitise
import string


def tpack(text, width=100):
    """Pack a list of words into lines, so long as each line (including
    intervening spaces) is no longer than _width_"""
    lines = [text[0]]
    for word in text[1:]:
        if len(lines[-1]) + 1 + len(word) <= width:
            lines[-1] += (' ' + word)
        else:
            lines += [word]
    return lines


def depunctuate_character(c):
    """Record the punctuation of a character"""
    if c in string.ascii_uppercase:
        return 'UPPER'
    elif c in string.ascii_lowercase:
        return 'LOWER'
    else:
        return c


def depunctuate(text):
    """Record the punctuation of a string, so it can be applied to a converted
    version of the string.

    For example, 
    punct = depunctuate(ciphertext)
    plaintext = decipher(sanitise(ciphertext))
    readable_plaintext = repunctuate(plaintext, punct)
    """
    return [depunctuate_character(c) for c in text]


def repunctuate_character(letters, punctuation):
    """Apply the recorded punctuation to a character. The letters must be
    an iterator of base characters."""
    if punctuation == 'UPPER':
        return next(letters).upper()
    elif punctuation == 'LOWER':
        return next(letters).lower()
    else:
        return punctuation


def repunctuate(text, punctuation):
    """Apply the recored punctuation to a sanitised string.

    For example, 
    punct = depunctuate(ciphertext)
    plaintext = decipher(sanitise(ciphertext))
    readable_plaintext = repunctuate(plaintext, punct)
    """
    letters = iter(sanitise(text))
    return cat(repunctuate_character(letters, p) for p in punctuation)

from enum import Enum
import multiprocessing 
import itertools

from support.utilities import *
from support.language_models import *
from cipher.column_transposition import transpositions, transpositions_of

from logger import logger

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
    fill_iterator = itertools.cycle(fillpattern)
    indices = itertools.count()
    message_length = len(message)

    current_position = 0
    grid = []
    current_fillpattern = fillpattern
    while current_position < message_length:
        row = []
        if fillstyle == AmscoFillStyle.same_each_row:
            fill_iterator = itertools.cycle(fillpattern)
        if fillstyle == AmscoFillStyle.reverse_each_row:
            fill_iterator = itertools.cycle(current_fillpattern)
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


def amsco_break(message, translist=transpositions, patterns = [(1, 2), (2, 1)],
                                  fillstyles = [AmscoFillStyle.continuous, 
                                                AmscoFillStyle.same_each_row, 
                                                AmscoFillStyle.reverse_each_row],
                                  fitness=Pbigrams, 
                                  chunksize=500):
    """Breaks an AMSCO transposition cipher using a dictionary and
    n-gram frequency analysis

    >>> amsco_break(amsco_transposition_encipher(sanitise( \
            "It is a truth universally acknowledged, that a single man in \
             possession of a good fortune, must be in want of a wife. However \
             little known the feelings or views of such a man may be on his \
             first entering a neighbourhood, this truth is so well fixed in \
             the minds of the surrounding families, that he is considered the \
             rightful property of some one or other of their daughters."), \
        'encipher'), \
        translist={(2, 0, 5, 3, 1, 4, 6): ['encipher'], \
                   (5, 0, 6, 1, 3, 4, 2): ['fourteen'], \
                   (6, 1, 0, 4, 5, 3, 2): ['keyword']}, \
        patterns=[(1, 2)]) # doctest: +ELLIPSIS
    (((2, 0, 5, 3, 1, 4, 6), (1, 2), <AmscoFillStyle.continuous: 1>), -709.4646722...)
    >>> amsco_break(amsco_transposition_encipher(sanitise( \
            "It is a truth universally acknowledged, that a single man in \
             possession of a good fortune, must be in want of a wife. However \
             little known the feelings or views of such a man may be on his \
             first entering a neighbourhood, this truth is so well fixed in \
             the minds of the surrounding families, that he is considered the \
             rightful property of some one or other of their daughters."), \
        'encipher', fillpattern=(2, 1)), \
        translist={(2, 0, 5, 3, 1, 4, 6): ['encipher'], \
                   (5, 0, 6, 1, 3, 4, 2): ['fourteen'], \
                   (6, 1, 0, 4, 5, 3, 2): ['keyword']}, \
        patterns=[(1, 2), (2, 1)], fitness=Ptrigrams) # doctest: +ELLIPSIS
    (((2, 0, 5, 3, 1, 4, 6), (2, 1), <AmscoFillStyle.continuous: 1>), -997.0129085...)
    """
    with multiprocessing.Pool() as pool:
        helper_args = [(message, trans, pattern, fillstyle, fitness)
                       for trans in translist
                       for pattern in patterns
                       for fillstyle in fillstyles]
        # Gotcha: the helper function here needs to be defined at the top level
        #   (limitation of Pool.starmap)
        breaks = pool.starmap(amsco_break_worker, helper_args, chunksize) 
        return max(breaks, key=lambda k: k[1])

def amsco_break_worker(message, transposition,
        pattern, fillstyle, fitness):
    plaintext = amsco_transposition_decipher(message, transposition,
        fillpattern=pattern, fillstyle=fillstyle)
    fit = fitness(sanitise(plaintext))
    logger.debug('AMSCO transposition break attempt using key {0} and pattern'
                         '{1} ({2}) gives fit of {3} and decrypt starting: '
                         '{4}'.format(
                             transposition, pattern, fillstyle, fit, 
                             sanitise(plaintext)[:50]))
    return (transposition, pattern, fillstyle), fit

if __name__ == "__main__":
    import doctest
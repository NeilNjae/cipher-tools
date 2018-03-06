import math
from enum import Enum
from itertools import starmap, zip_longest
from support.utilities import *
from support.language_models import *


from logger import logger

def railfence_encipher(message, height, fillvalue=''):
    """Railfence cipher.
    Works by splitting the text into sections, then reading across them to
    generate the rows in the cipher. The rows are then combined to form the
    ciphertext.

    Example: the plaintext "hellotherefriends", with a height of four, written 
    out in the railfence as 
       h h i
       etere*
       lorfns
       l e d
    (with the * showing the one character to finish the last section). 
    Each 'section' is two columns, but unfolded. In the example, the first
    section is 'hellot'.

    >>> railfence_encipher('hellothereavastmeheartiesthisisalongpieceoftextfortestingrailfenceciphers', 2, fillvalue='!')
    'hlohraateerishsslnpeefetotsigaleccpeselteevsmhatetiiaogicotxfretnrifneihr!'
    >>> railfence_encipher('hellothereavastmeheartiesthisisalongpieceoftextfortestingrailfenceciphers', 3, fillvalue='!')
    'horaersslpeeosglcpselteevsmhatetiiaogicotxfretnrifneihr!!lhateihsnefttiaece!'
    >>> railfence_encipher('hellothereavastmeheartiesthisisalongpieceoftextfortestingrailfenceciphers', 5, fillvalue='!')
    'hresleogcseeemhetaocofrnrner!!lhateihsnefttiaece!!ltvsatiigitxetifih!!oarspeslp!'
    >>> railfence_encipher('hellothereavastmeheartiesthisisalongpieceoftextfortestingrailfenceciphers', 10, fillvalue='!')
    'hepisehagitnr!!lernesge!!lmtocerh!!otiletap!!tseaorii!!hassfolc!!evtitffe!!rahsetec!!eixn!'
    >>> railfence_encipher('hellothereavastmeheartiesthisisalongpieceoftextfortestingrailfenceciphers', 3)
    'horaersslpeeosglcpselteevsmhatetiiaogicotxfretnrifneihrlhateihsnefttiaece'
    >>> railfence_encipher('hellothereavastmeheartiesthisisalongpieceoftextfortestingrailfenceciphers', 5)
    'hresleogcseeemhetaocofrnrnerlhateihsnefttiaeceltvsatiigitxetifihoarspeslp'
    >>> railfence_encipher('hellothereavastmeheartiesthisisalongpieceoftextfortestingrailfenceciphers', 7)
    'haspolsevsetgifrifrlatihnettaeelemtiocxernhorersleesgcptehaiaottneihesfic'
    """
    sections = chunks(message, (height - 1) * 2, fillvalue=fillvalue)
    n_sections = len(sections)
    # Add the top row
    rows = [cat([s[0] for s in sections])]
    # process the middle rows of the grid
    for r in range(1, height-1):
        rows += [cat([s[r:r+1] + s[height*2-r-2:height*2-r-1] for s in sections])]
    # process the bottom row
    rows += [cat([s[height - 1:height] for s in sections])]
    # rows += [wcat([s[height - 1] for s in sections])]
    return cat(rows)

def railfence_decipher(message, height, fillvalue=''):
    """Railfence decipher. 
    Works by reconstructing the grid used to generate the ciphertext, then
    unfolding the sections so the text can be concatenated together.

    Example: given the ciphertext 'hhieterelorfnsled' and a height of 4, first
    work out that the second row has a character missing, find the rows of the
    grid, then split the section into its two columns.

    'hhieterelorfnsled' is split into
        h h i
        etere
        lorfns
        l e d
    (spaces added for clarity), which is stored in 'rows'. This is then split
    into 'down_rows' and 'up_rows':

    down_rows:
       hhi
       eee
       lrn
       led

    up_rows:
       tr
       ofs

    These are then zipped together (after the up_rows are reversed) to recover 
    the plaintext.

    Most of the procedure is about finding the correct lengths for each row then
    splitting the ciphertext into those rows.

    >>> railfence_decipher('hlohraateerishsslnpeefetotsigaleccpeselteevsmhatetiiaogicotxfretnrifneihr!', 2).strip('!')
    'hellothereavastmeheartiesthisisalongpieceoftextfortestingrailfenceciphers'
    >>> railfence_decipher('horaersslpeeosglcpselteevsmhatetiiaogicotxfretnrifneihr!!lhateihsnefttiaece!', 3).strip('!')
    'hellothereavastmeheartiesthisisalongpieceoftextfortestingrailfenceciphers'
    >>> railfence_decipher('hresleogcseeemhetaocofrnrner!!lhateihsnefttiaece!!ltvsatiigitxetifih!!oarspeslp!', 5).strip('!')
    'hellothereavastmeheartiesthisisalongpieceoftextfortestingrailfenceciphers'
    >>> railfence_decipher('hepisehagitnr!!lernesge!!lmtocerh!!otiletap!!tseaorii!!hassfolc!!evtitffe!!rahsetec!!eixn!', 10).strip('!')
    'hellothereavastmeheartiesthisisalongpieceoftextfortestingrailfenceciphers'
    >>> railfence_decipher('horaersslpeeosglcpselteevsmhatetiiaogicotxfretnrifneihrlhateihsnefttiaece', 3)
    'hellothereavastmeheartiesthisisalongpieceoftextfortestingrailfenceciphers'
    >>> railfence_decipher('hresleogcseeemhetaocofrnrnerlhateihsnefttiaeceltvsatiigitxetifihoarspeslp', 5)
    'hellothereavastmeheartiesthisisalongpieceoftextfortestingrailfenceciphers'
    >>> railfence_decipher('haspolsevsetgifrifrlatihnettaeelemtiocxernhorersleesgcptehaiaottneihesfic', 7)
    'hellothereavastmeheartiesthisisalongpieceoftextfortestingrailfenceciphers'
    """
    # find the number and size of the sections, including how many characters
    #   are missing for a full grid
    n_sections = math.ceil(len(message) / ((height - 1) * 2))
    padding_to_add = n_sections * (height - 1) * 2 - len(message)
    # row_lengths are for the both up rows and down rows
    row_lengths = [n_sections] * (height - 1) * 2
    for i in range((height - 1) * 2 - 1, (height - 1) * 2 - (padding_to_add + 1), -1):
        row_lengths[i] -= 1
    # folded_rows are the combined row lengths in the middle of the railfence
    folded_row_lengths = [row_lengths[0]]
    for i in range(1, height-1):
        folded_row_lengths += [row_lengths[i] + row_lengths[-i]]
    folded_row_lengths += [row_lengths[height - 1]]
    # find the rows that form the railfence grid
    rows = []
    row_start = 0
    for i in folded_row_lengths:
        rows += [message[row_start:row_start + i]]
        row_start += i
    # split the rows into the 'down_rows' (those that form the first column of
    #   a section) and the 'up_rows' (those that ofrm the second column of a 
    #   section).
    down_rows = [rows[0]]
    up_rows = []
    for i in range(1, height-1):
        down_rows += [cat([c for n, c in enumerate(rows[i]) if n % 2 == 0])]
        up_rows += [cat([c for n, c in enumerate(rows[i]) if n % 2 == 1])]
    down_rows += [rows[-1]]
    up_rows.reverse()
    return cat(c for r in zip_longest(*(down_rows + up_rows), fillvalue='') for c in r)


def railfence_break(message, max_key_length=20,
                     fitness=Pletters, chunksize=500):
    """Breaks a railfence cipher using a matrix of given rank and letter frequencies

    
    """
    
    sanitised_message = sanitise(message)
    results = starmap(worker, [(sanitised_message, i, fitness)
                               for i in range(2, max_key_length+1)])
    return max(results, key=lambda k: k[1])


def railfence_break(message, max_key_length=20,
                     fitness=Pbigrams, chunksize=500):
    """Breaks a railfence cipher using a range of lengths and
    n-gram frequency analysis

    >>> railfence_break(railfence_encipher(sanitise( \
            "It is a truth universally acknowledged, that a single man in \
             possession of a good fortune, must be in want of a wife. However \
             little known the feelings or views of such a man may be on his \
             first entering a neighbourhood, this truth is so well fixed in \
             the minds of the surrounding families, that he is considered the \
             rightful property of some one or other of their daughters."), \
        7)) # doctest: +ELLIPSIS
    (7, -709.46467226...)
    >>> railfence_break(railfence_encipher(sanitise( \
            "It is a truth universally acknowledged, that a single man in \
             possession of a good fortune, must be in want of a wife. However \
             little known the feelings or views of such a man may be on his \
             first entering a neighbourhood, this truth is so well fixed in \
             the minds of the surrounding families, that he is considered the \
             rightful property of some one or other of their daughters."), \
        7), \
        fitness=Ptrigrams) # doctest: +ELLIPSIS
    (7, -997.0129085...)
    """
    def worker(message, height, fitness):
        plaintext = railfence_decipher(message, height)
        fit = fitness(plaintext)
        return height, fit

    sanitised_message = sanitise(message)
    results = starmap(worker, [(sanitised_message, i, fitness)
                               for i in range(2, max_key_length+1)])
    return max(results, key=lambda k: k[1])

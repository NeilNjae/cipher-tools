from itertools import chain
import multiprocessing
from support.utilities import *
from support.language_models import *
from cipher.column_transposition import transpositions_of

from logger import logger

def make_cadenus_keycolumn(doubled_letters = 'vw', start='a', reverse=False):
    """Makes the key column for a Cadenus cipher (the column down between the
        rows of letters)

    >>> make_cadenus_keycolumn()['a']
    0
    >>> make_cadenus_keycolumn()['b']
    1
    >>> make_cadenus_keycolumn()['c']
    2
    >>> make_cadenus_keycolumn()['v']
    21
    >>> make_cadenus_keycolumn()['w']
    21
    >>> make_cadenus_keycolumn()['z']
    24
    >>> make_cadenus_keycolumn(doubled_letters='ij', start='b', reverse=True)['a']
    1
    >>> make_cadenus_keycolumn(doubled_letters='ij', start='b', reverse=True)['b']
    0
    >>> make_cadenus_keycolumn(doubled_letters='ij', start='b', reverse=True)['c']
    24
    >>> make_cadenus_keycolumn(doubled_letters='ij', start='b', reverse=True)['i']
    18
    >>> make_cadenus_keycolumn(doubled_letters='ij', start='b', reverse=True)['j']
    18
    >>> make_cadenus_keycolumn(doubled_letters='ij', start='b', reverse=True)['v']
    6
    >>> make_cadenus_keycolumn(doubled_letters='ij', start='b', reverse=True)['z']
    2
    """
    index_to_remove = string.ascii_lowercase.find(doubled_letters[0])
    short_alphabet = string.ascii_lowercase[:index_to_remove] + string.ascii_lowercase[index_to_remove+1:]
    if reverse:
        short_alphabet = cat(reversed(short_alphabet))
    start_pos = short_alphabet.find(start)
    rotated_alphabet = short_alphabet[start_pos:] + short_alphabet[:start_pos]
    keycolumn = {l: i for i, l in enumerate(rotated_alphabet)}
    keycolumn[doubled_letters[0]] = keycolumn[doubled_letters[1]]
    return keycolumn

def cadenus_encipher(message, keyword, keycolumn, fillvalue='a'):
    """Encipher with the Cadenus cipher

    >>> cadenus_encipher(sanitise('Whoever has made a voyage up the Hudson ' \
                                  'must remember the Kaatskill mountains. ' \
                                  'They are a dismembered branch of the great'), \
                'wink', \
                make_cadenus_keycolumn(doubled_letters='vw', start='a', reverse=True))
    'antodeleeeuhrsidrbhmhdrrhnimefmthgeaetakseomehetyaasuvoyegrastmmuuaeenabbtpchehtarorikswosmvaleatned'
    >>> cadenus_encipher(sanitise('a severe limitation on the usefulness of ' \
                                  'the cadenus is that every message must be ' \
                                  'a multiple of twenty-five letters long'), \
                'easy', \
                make_cadenus_keycolumn(doubled_letters='vw', start='a', reverse=True))
    'systretomtattlusoatleeesfiyheasdfnmschbhneuvsnpmtofarenuseieeieltarlmentieetogevesitfaisltngeeuvowul'
    """
    rows = chunks(message, len(message) // 25, fillvalue=fillvalue)
    columns = zip(*rows)
    rotated_columns = [col[start:] + col[:start] for start, col in zip([keycolumn[l] for l in keyword], columns)]    
    rotated_rows = zip(*rotated_columns)
    transpositions = transpositions_of(keyword)
    transposed = [transpose(r, transpositions) for r in rotated_rows]
    return cat(chain(*transposed))

def cadenus_decipher(message, keyword, keycolumn, fillvalue='a'):
    """
    >>> cadenus_decipher('antodeleeeuhrsidrbhmhdrrhnimefmthgeaetakseomehetyaa' \
                         'suvoyegrastmmuuaeenabbtpchehtarorikswosmvaleatned', \
                 'wink', \
                 make_cadenus_keycolumn(reverse=True))
    'whoeverhasmadeavoyageupthehudsonmustrememberthekaatskillmountainstheyareadismemberedbranchofthegreat'
    >>> cadenus_decipher('systretomtattlusoatleeesfiyheasdfnmschbhneuvsnpmtof' \
                        'arenuseieeieltarlmentieetogevesitfaisltngeeuvowul', \
                 'easy', \
                 make_cadenus_keycolumn(reverse=True))
    'aseverelimitationontheusefulnessofthecadenusisthateverymessagemustbeamultipleoftwentyfiveletterslong'
    """
    rows = chunks(message, len(message) // 25, fillvalue=fillvalue)
    transpositions = transpositions_of(keyword)
    untransposed_rows = [untranspose(r, transpositions) for r in rows]
    columns = zip(*untransposed_rows)
    rotated_columns = [col[-start:] + col[:-start] for start, col in zip([keycolumn[l] for l in keyword], columns)]    
    rotated_rows = zip(*rotated_columns)
    # return rotated_columns
    return cat(chain(*rotated_rows))


def cadenus_break(message, words=keywords, 
    doubled_letters='vw', fitness=Pbigrams):
    c = make_cadenus_keycolumn(reverse=True)
    valid_words = [w for w in words 
        if max(transpositions_of(w)) <= len(c)]
    with multiprocessing.Pool() as pool:
        results = pool.starmap(cadenus_break_worker, 
                [(message, w, 
                    make_cadenus_keycolumn(doubled_letters=doubled_letters, 
                        start=s, reverse=r), 
                    fitness)
                for w in words 
                for s in string.ascii_lowercase 
                for r in [True, False]
                if max(transpositions_of(w)) <= len(
                    make_cadenus_keycolumn(
                        doubled_letters=doubled_letters, start=s, reverse=r))
                ])
    # return list(results)
    return max(results, key=lambda k: k[1])

def cadenus_break_worker(message, keyword, keycolumn, fitness):
    message_chunks = chunks(message, 175)
    plaintext = ''.join(cadenus_decipher(c, keyword, keycolumn) for c in message_chunks)
    fit = fitness(plaintext)
    return (keyword, keycolumn), fit

if __name__ == "__main__":
    import doctest
import multiprocessing
import numpy as np
from numpy import matrix
from numpy import linalg
from support.utilities import *
from support.language_models import *
from cipher.affine import modular_division_table

from logger import logger


def hill_encipher(matrix, message_letters, fillvalue='a'):
    """Hill cipher

    >>> hill_encipher(np.matrix([[7,8], [11,11]]), 'hellothere')
    'drjiqzdrvx'
    >>> hill_encipher(np.matrix([[6, 24, 1], [13, 16, 10], [20, 17, 15]]), \
        'hello there')
    'tfjflpznvyac'
    """
    n = len(matrix)
    sanitised_message = sanitise(message_letters)
    if len(sanitised_message) % n != 0:
        padding = fillvalue[0] * (n - len(sanitised_message) % n)
    else:
        padding = ''
    message = [pos(c) for c in sanitised_message + padding]
    message_chunks = [message[i:i+n] for i in range(0, len(message), n)]
    # message_chunks = chunks(message, len(matrix), fillvalue=None)
    enciphered_chunks = [((matrix * np.matrix(c).T).T).tolist()[0] 
            for c in message_chunks]
    return cat([unpos(round(l))
            for l in sum(enciphered_chunks, [])])

def hill_decipher(matrix, message, fillvalue='a'):
    """Hill cipher

    >>> hill_decipher(np.matrix([[7,8], [11,11]]), 'drjiqzdrvx')
    'hellothere'
    >>> hill_decipher(np.matrix([[6, 24, 1], [13, 16, 10], [20, 17, 15]]), \
        'tfjflpznvyac')
    'hellothereaa'
    """
    adjoint = linalg.det(matrix)*linalg.inv(matrix)
    inverse_determinant = modular_division_table[int(round(linalg.det(matrix))) % 26, 1]
    inverse_matrix = (inverse_determinant * adjoint) % 26
    return hill_encipher(inverse_matrix, message, fillvalue)          

def hill_break(message, matrix_size=2, fitness=Pletters, 
    number_of_solutions=1, chunksize=500):

    all_matrices = [np.matrix(list(m)) 
        for m in itertools.product([list(r) 
            for r in itertools.product(range(26), repeat=matrix_size)], 
        repeat=matrix_size)]
    valid_matrices = [m for m, d in 
        zip(all_matrices, (int(round(linalg.det(m))) for m in all_matrices))
                  if d != 0
                  if d % 2 != 0
                  if d % 13 != 0 ]
    with multiprocessing.Pool() as pool:
        helper_args = [(message, matrix, fitness)
                       for matrix in valid_matrices]
        # Gotcha: the helper function here needs to be defined at the top level
        #   (limitation of Pool.starmap)
        breaks = pool.starmap(hill_break_worker, helper_args, chunksize)
        if number_of_solutions == 1:
            return max(breaks, key=lambda k: k[1])
        else:
            return sorted(breaks, key=lambda k: k[1], reverse=True)[:number_of_solutions]

def hill_break_worker(message, matrix, fitness):
    plaintext = hill_decipher(matrix, message)
    fit = fitness(plaintext)
    logger.debug('Hill cipher break attempt using key {0} gives fit of '
                 '{1} and decrypt starting: {2}'.format(matrix, 
                     fit, sanitise(plaintext)[:50]))
    return matrix, fit

if __name__ == "__main__":
    import doctest
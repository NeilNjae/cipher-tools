import math
import multiprocessing 
from support.utilities import *
from support.language_models import *
from cipher.caesar import caesar_encipher_letter, caesar_decipher_letter

from logger import logger


def autokey_encipher(message, keyword):
    """Encipher with the autokey cipher

    >>> autokey_encipher('meetatthefountain', 'kilt')
    'wmpmmxxaeyhbryoca'
    """
    shifts = [pos(l) for l in keyword + message]
    pairs = zip(message, shifts)
    return cat([caesar_encipher_letter(l, k) for l, k in pairs])

def autokey_decipher(ciphertext, keyword):
    """Decipher with the autokey cipher

    >>> autokey_decipher('wmpmmxxaeyhbryoca', 'kilt')
    'meetatthefountain'
    """
    plaintext = []
    keys = list(keyword)
    for c in ciphertext:
        plaintext_letter = caesar_decipher_letter(c, pos(keys[0]))
        plaintext += [plaintext_letter]
        keys = keys[1:] + [plaintext_letter]
    return cat(plaintext)



def autokey_sa_break( message
                    , min_keylength=2
                    , max_keylength=20
                    , workers=10
                    , initial_temperature=200
                    , max_iterations=20000
                    , fitness=Pletters
                    , chunksize=1
                    , result_count=1
                    ):
    """Break an autokey cipher by simulated annealing
    """
    worker_args = []
    ciphertext = sanitise(message)
    for keylength in range(min_keylength, max_keylength+1):
        for i in range(workers):
            key = cat(random.choice(string.ascii_lowercase) for _ in range(keylength))
            worker_args.append((ciphertext, key, 
                            initial_temperature, max_iterations, fitness))
            
    with multiprocessing.Pool() as pool:
        breaks = pool.starmap(autokey_sa_break_worker,
                              worker_args, chunksize)
    if result_count <= 1:
        return max(breaks, key=lambda k: k[1])
    else:
        return sorted(set(breaks), key=lambda k: k[1], reverse=True)[:result_count]


def autokey_sa_break_worker(message, key, 
                                     t0, max_iterations, fitness):
   
    temperature = t0

    dt = t0 / (0.9 * max_iterations)
    
    plaintext = autokey_decipher(message, key)
    current_fitness = fitness(plaintext)
    current_key = key

    best_key = current_key
    best_fitness = current_fitness
    best_plaintext = plaintext
    
    # print('starting for', max_iterations)
    for i in range(max_iterations):
        swap_pos = random.randrange(len(current_key))
        swap_char = random.choice(string.ascii_lowercase)
        
        new_key = current_key[:swap_pos] + swap_char + current_key[swap_pos+1:]
        
        plaintext = autokey_decipher(message, new_key)
        new_fitness = fitness(plaintext)
        try:
            sa_chance = math.exp((new_fitness - current_fitness) / temperature)
        except (OverflowError, ZeroDivisionError):
            # print('exception triggered: new_fit {}, current_fit {}, temp {}'.format(new_fitness, current_fitness, temperature))
            sa_chance = 0
        if (new_fitness > current_fitness or random.random() < sa_chance):
            # logger.debug('Simulated annealing: iteration {}, temperature {}, '
            #     'current alphabet {}, current_fitness {}, '
            #     'best_plaintext {}'.format(i, temperature, current_alphabet, 
            #     current_fitness, best_plaintext[:50]))

            # logger.debug('new_fit {}, current_fit {}, temp {}, sa_chance {}'.format(new_fitness, current_fitness, temperature, sa_chance))
#             print(new_fitness, new_key, plaintext[:100])
            current_fitness = new_fitness
            current_key = new_key
            
        if current_fitness > best_fitness:
            best_key = current_key
            best_fitness = current_fitness
            best_plaintext = plaintext
        if i % 500 == 0:
            logger.debug('Simulated annealing: iteration {}, temperature {}, '
                'current key {}, current_fitness {}, '
                'best_plaintext {}'.format(i, temperature, current_key, 
                current_fitness, plaintext[:50]))
        temperature = max(temperature - dt, 0.001)
        
#     print(best_key, best_fitness, best_plaintext[:70])
    return best_key, best_fitness # current_alphabet, current_fitness

if __name__ == "__main__":
    import doctest
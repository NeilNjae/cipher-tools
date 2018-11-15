from support.utilities import *
from support.language_models import *
from cipher.keyword_cipher import KeywordWrapAlphabet, keyword_cipher_alphabet_of
from cipher.polybius import polybius_grid
import multiprocessing

from logger import logger

def playfair_wrap(n, lowest, highest):
    skip = highest - lowest + 1
    while n > highest or n < lowest:
        if n > highest:
            n -= skip
        if n < lowest:
            n += skip
    return n

def playfair_encipher_bigram(ab, grid, padding_letter='x'):
    a, b = ab
    max_row = max(c[0] for c in grid.values())
    max_col = max(c[1] for c in grid.values())
    min_row = min(c[0] for c in grid.values())
    min_col = min(c[1] for c in grid.values())
    if a == b:
        b = padding_letter
    if grid[a][0] == grid[b][0]:  # same row
        cp = (grid[a][0], playfair_wrap(grid[a][1] + 1, min_col, max_col))
        dp = (grid[b][0], playfair_wrap(grid[b][1] + 1, min_col, max_col))
    elif grid[a][1] == grid[b][1]:  # same column
        cp = (playfair_wrap(grid[a][0] + 1, min_row, max_row), grid[a][1])
        dp = (playfair_wrap(grid[b][0] + 1, min_row, max_row), grid[b][1])
    else:
        cp = (grid[a][0], grid[b][1])
        dp = (grid[b][0], grid[a][1])
    c = [k for k, v in grid.items() if v == cp][0]
    d = [k for k, v in grid.items() if v == dp][0]
    return c + d

def playfair_decipher_bigram(ab, grid, padding_letter='x'):
    a, b = ab
    max_row = max(c[0] for c in grid.values())
    max_col = max(c[1] for c in grid.values())
    min_row = min(c[0] for c in grid.values())
    min_col = min(c[1] for c in grid.values())
    if a == b:
        b = padding_letter
    if grid[a][0] == grid[b][0]:  # same row
        cp = (grid[a][0], playfair_wrap(grid[a][1] - 1, min_col, max_col))
        dp = (grid[b][0], playfair_wrap(grid[b][1] - 1, min_col, max_col))
    elif grid[a][1] == grid[b][1]:  # same column
        cp = (playfair_wrap(grid[a][0] - 1, min_row, max_row), grid[a][1])
        dp = (playfair_wrap(grid[b][0] - 1, min_row, max_row), grid[b][1])
    else:
        cp = (grid[a][0], grid[b][1])
        dp = (grid[b][0], grid[a][1])
    c = [k for k, v in grid.items() if v == cp][0]
    d = [k for k, v in grid.items() if v == dp][0]
    return c + d

def playfair_bigrams(text, padding_letter='x', padding_replaces_repeat=True):
    i = 0
    bigrams = []
    while i < len(text):
        bigram = text[i:i+2]
        if len(bigram) == 1:
            i = len(text) + 1
            bigram = bigram + padding_letter
        else:
            if bigram[0] == bigram[1]:
                bigram = bigram[0] + padding_letter
                if padding_replaces_repeat:
                    i += 2
                else:
                    i += 1
            else:
                i += 2
        bigrams += [bigram]
    return bigrams

def playfair_encipher(message, keyword, padding_letter='x',
                      padding_replaces_repeat=False, letters_to_merge=None, 
                      wrap_alphabet=KeywordWrapAlphabet.from_a):
    column_order = list(range(5))
    row_order = list(range(5))
    if letters_to_merge is None: 
        letters_to_merge = {'j': 'i'}   
    grid = polybius_grid(keyword, column_order, row_order,
                        letters_to_merge=letters_to_merge,
                        wrap_alphabet=wrap_alphabet)
    message_bigrams = playfair_bigrams(sanitise(message), padding_letter=padding_letter, 
                                       padding_replaces_repeat=padding_replaces_repeat)
    ciphertext_bigrams = [playfair_encipher_bigram(b, grid, padding_letter=padding_letter) for b in message_bigrams]
    return cat(ciphertext_bigrams)

def playfair_decipher(message, keyword, padding_letter='x',
                      padding_replaces_repeat=False, letters_to_merge=None, 
                      wrap_alphabet=KeywordWrapAlphabet.from_a):
    column_order = list(range(5))
    row_order = list(range(5))
    if letters_to_merge is None: 
        letters_to_merge = {'j': 'i'}   
    grid = polybius_grid(keyword, column_order, row_order,
                        letters_to_merge=letters_to_merge,
                        wrap_alphabet=wrap_alphabet)
    message_bigrams = playfair_bigrams(sanitise(message), padding_letter=padding_letter, 
                                       padding_replaces_repeat=padding_replaces_repeat)
    plaintext_bigrams = [playfair_decipher_bigram(b, grid, padding_letter=padding_letter) for b in message_bigrams]
    return cat(plaintext_bigrams)

def playfair_break_mp(message, 
                      letters_to_merge=None, padding_letter='x',
                      wordlist=keywords, fitness=Pletters,
                      number_of_solutions=1, chunksize=500):
    if letters_to_merge is None: 
        letters_to_merge = {'j': 'i'}   

    with multiprocessing.Pool() as pool:
        helper_args = [(message, word, wrap, 
                        letters_to_merge, padding_letter,
                        pad_replace,
                        fitness)
                       for word in wordlist
                       for wrap in KeywordWrapAlphabet
                       for pad_replace in [False, True]]
        # Gotcha: the helper function here needs to be defined at the top level
        #   (limitation of Pool.starmap)
        breaks = pool.starmap(playfair_break_worker, helper_args, chunksize)
        if number_of_solutions == 1:
            return max(breaks, key=lambda k: k[1])
        else:
            return sorted(breaks, key=lambda k: k[1], reverse=True)[:number_of_solutions]

def playfair_break_worker(message, keyword, wrap, 
                          letters_to_merge, padding_letter,
                          pad_replace,
                          fitness):
    plaintext = playfair_decipher(message, keyword, padding_letter,
                                  pad_replace,
                                  letters_to_merge, 
                                  wrap)
    if plaintext:
        fit = fitness(plaintext)
    else:
        fit = float('-inf')
    logger.debug('Playfair break attempt using key {0} (wrap={1}, merging {2}, '
                 'pad replaces={3}), '
                 'gives fit of {4} and decrypt starting: '
                 '{5}'.format(keyword, wrap, letters_to_merge, pad_replace,
                              fit, sanitise(plaintext)[:50]))
    return (keyword, wrap, letters_to_merge, padding_letter, pad_replace), fit

def playfair_simulated_annealing_break(message, workers=10, 
                              initial_temperature=200,
                              max_iterations=20000,
                              plain_alphabet=None, 
                              cipher_alphabet=None, 
                              fitness=Pletters, chunksize=1):
    worker_args = []
    ciphertext = sanitise(message)
    for i in range(workers):
        if plain_alphabet is None:
            used_plain_alphabet = string.ascii_lowercase
        else:
            used_plain_alphabet = plain_alphabet
        if cipher_alphabet is None:
            # used_cipher_alphabet = list(string.ascii_lowercase)
            # random.shuffle(used_cipher_alphabet)
            # used_cipher_alphabet = cat(used_cipher_alphabet)
            used_cipher_alphabet = random.choice(keywords)
        else:
            used_cipher_alphabet = cipher_alphabet
        worker_args.append((ciphertext, used_plain_alphabet, used_cipher_alphabet, 
                            initial_temperature, max_iterations, fitness))
    with multiprocessing.Pool() as pool:
        breaks = pool.starmap(playfair_simulated_annealing_break_worker,
                              worker_args, chunksize)
    return max(breaks, key=lambda k: k[1])

def playfair_simulated_annealing_break_worker(message, plain_alphabet, cipher_alphabet, 
                                     t0, max_iterations, fitness):
    def swap(letters, i, j):
        if i > j:
            i, j = j, i
        if i == j:
            return letters
        else:
            return (letters[:i] + letters[j] + letters[i+1:j] + letters[i] +
                    letters[j+1:])
    
    temperature = t0

    dt = t0 / (0.9 * max_iterations)
    
    current_alphabet = cipher_alphabet
    current_wrap = KeywordWrapAlphabet.from_a
    current_letters_to_merge = {'j': 'i'}
    current_pad_replace = False
    current_padding_letter = 'x'
    
    alphabet = current_alphabet
    wrap = current_wrap
    letters_to_merge = current_letters_to_merge
    pad_replace = current_pad_replace
    padding_letter = current_padding_letter
    plaintext = playfair_decipher(message, alphabet, padding_letter,
                                  pad_replace,
                                  letters_to_merge, 
                                  wrap)
    current_fitness = fitness(plaintext)

    best_alphabet = current_alphabet
    best_fitness = current_fitness
    best_plaintext = plaintext
    
    # print('starting for', max_iterations)
    for i in range(max_iterations):
        chosen = random.random()
        # if chosen < 0.7:
        #     swap_a = random.randrange(26)
        #     swap_b = (swap_a + int(random.gauss(0, 4))) % 26
        #     alphabet = swap(current_alphabet, swap_a, swap_b)
        # elif chosen < 0.8:
        #     wrap = random.choice(list(KeywordWrapAlphabet))
        # elif chosen < 0.9:
        #     pad_replace = random.choice([True, False])
        # elif chosen < 0.95:
        #     letter_from = random.choice(string.ascii_lowercase)
        #     letter_to = random.choice([c for c in string.ascii_lowercase if c != letter_from])
        #     letters_to_merge = {letter_from: letter_to}
        # else:
        #     padding_letter = random.choice(string.ascii_lowercase)
        if chosen < 0.7:
            swap_a = random.randrange(len(current_alphabet))
            swap_b = (swap_a + int(random.gauss(0, 4))) % len(current_alphabet)
            alphabet = swap(current_alphabet, swap_a, swap_b)
        elif chosen < 0.85:
            new_letter = random.choice(string.ascii_lowercase)
            alphabet = swap(current_alphabet + new_letter, random.randrange(len(current_alphabet)), len(current_alphabet))
        else:
            if len(current_alphabet) > 1:
                deletion_position = random.randrange(len(current_alphabet))
                alphabet = current_alphabet[:deletion_position] + current_alphabet[deletion_position+1:]
            else:
                alphabet = current_alphabet
        
        try:
            plaintext = playfair_decipher(message, alphabet, padding_letter,
                                  pad_replace,
                                  letters_to_merge, 
                                  wrap)
        except:
            print("Error", alphabet, padding_letter,
                                  pad_replace,
                                  letters_to_merge, 
                                  wrap)
            raise

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
            current_fitness = new_fitness
            current_alphabet = alphabet
            current_wrap = wrap
            current_letters_to_merge = letters_to_merge
            current_pad_replace = pad_replace
            current_padding_letter = padding_letter
            
        if current_fitness > best_fitness:
            best_alphabet = current_alphabet
            best_wrap = current_wrap
            best_letters_to_merge = current_letters_to_merge
            best_pad_replace = current_pad_replace
            best_padding_letter = current_padding_letter
            best_fitness = current_fitness
            best_plaintext = plaintext
        if i % 500 == 0:
            logger.debug('Simulated annealing: iteration {}, temperature {}, '
                'current alphabet {}, current_fitness {}, '
                'best_plaintext {}'.format(i, temperature, current_alphabet, 
                current_fitness, plaintext[:50]))
        temperature = max(temperature - dt, 0.001)

    return { 'alphabet': best_alphabet
           , 'wrap': best_wrap
           , 'letters_to_merge': best_letters_to_merge
           , 'pad_replace': best_pad_replace
           , 'padding_letter': best_padding_letter
           }, best_fitness # current_alphabet, current_fitness

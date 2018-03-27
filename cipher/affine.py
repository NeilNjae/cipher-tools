from support.utilities import *
from support.language_models import *
from logger import logger


modular_division_table = {
    (multiplier, (multiplier * plaintext) % 26): plaintext
    for plaintext in range(26) 
    for multiplier in range(26)
    }


def affine_encipher_letter(accented_letter, multiplier=1, adder=0, one_based=True):
    """Encipher a letter, given a multiplier and adder
    
    >>> cat(affine_encipher_letter(l, 3, 5, True) \
            for l in string.ascii_letters)
    'hknqtwzcfiloruxadgjmpsvybeHKNQTWZCFILORUXADGJMPSVYBE'
    >>> cat(affine_encipher_letter(l, 3, 5, False) \
            for l in string.ascii_letters)
    'filoruxadgjmpsvybehknqtwzcFILORUXADGJMPSVYBEHKNQTWZC'
    """
    letter = unaccent(accented_letter)
    if letter in string.ascii_letters:
        letter_number = pos(letter)
        if one_based: letter_number += 1
        cipher_number = (letter_number * multiplier + adder) % 26
        if one_based: cipher_number -= 1
        if letter in string.ascii_uppercase:
            return unpos(cipher_number).upper()
        else:
            return unpos(cipher_number)
    else:
        return letter

def affine_decipher_letter(letter, multiplier=1, adder=0, one_based=True):
    """Encipher a letter, given a multiplier and adder
    
    >>> cat(affine_decipher_letter(l, 3, 5, True) \
            for l in 'hknqtwzcfiloruxadgjmpsvybeHKNQTWZCFILORUXADGJMPSVYBE')
    'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    >>> cat(affine_decipher_letter(l, 3, 5, False) \
            for l in 'filoruxadgjmpsvybehknqtwzcFILORUXADGJMPSVYBEHKNQTWZC')
    'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    """
    if letter in string.ascii_letters:
        cipher_number = pos(letter)
        if one_based: cipher_number += 1
        # plaintext_number = ( 
        #     modular_division_table[multiplier]
        #                           [(cipher_number - adder) % 26])
        plaintext_number = ( 
            modular_division_table[multiplier, (cipher_number - adder) % 26]
            )
        if one_based: plaintext_number -= 1
        if letter in string.ascii_uppercase:
            return unpos(plaintext_number).upper()
        else:
            return unpos(plaintext_number) 
    else:
        return letter

def affine_encipher(message, multiplier=1, adder=0, one_based=True):
    """Encipher a message
    
    >>> affine_encipher('hours passed during which jerico tried every ' \
           'trick he could think of', 15, 22, True)
    'lmyfu bkuusd dyfaxw claol psfaom jfasd snsfg jfaoe ls omytd jlaxe mh'
    """
    enciphered = [affine_encipher_letter(l, multiplier, adder, one_based) 
                  for l in message]
    return cat(enciphered)

def affine_decipher(message, multiplier=1, adder=0, one_based=True):
    """Decipher a message
    
    >>> affine_decipher('lmyfu bkuusd dyfaxw claol psfaom jfasd snsfg ' \
           'jfaoe ls omytd jlaxe mh', 15, 22, True)
    'hours passed during which jerico tried every trick he could think of'
    """
    enciphered = [affine_decipher_letter(l, multiplier, adder, one_based) 
                  for l in message]
    return cat(enciphered)



def affine_break(message, fitness=Pletters):
    """Breaks an affine cipher using frequency analysis

    >>> affine_break('lmyfu bkuusd dyfaxw claol psfaom jfasd snsfg jfaoe ls ' \
          'omytd jlaxe mh jm bfmibj umis hfsul axubafkjamx. ls kffkxwsd jls ' \
          'ofgbjmwfkiu olfmxmtmwaokttg jlsx ls kffkxwsd jlsi zg tsxwjl. jlsx ' \
          'ls umfjsd jlsi zg hfsqysxog. ls dmmdtsd mx jls bats mh bkbsf. ls ' \
          'bfmctsd kfmyxd jls lyj, mztanamyu xmc jm clm cku tmmeaxw kj lai ' \
          'kxd clm ckuxj.') # doctest: +ELLIPSIS
    ((15, 22, True), -340.601181913...)
    """
    sanitised_message = sanitise(message)
    best_multiplier = 0
    best_adder = 0
    best_one_based = True
    best_fit = float("-inf")
    for one_based in [True, False]:
        for multiplier in [x for x in range(1, 26, 2) if x != 13]:
            for adder in range(26):
                plaintext = affine_decipher(sanitised_message,
                                            multiplier, adder, one_based)
                fit = fitness(plaintext)
                logger.debug('Affine break attempt using key {0}x+{1} ({2}) '
                             'gives fit of {3} and decrypt starting: {4}'.
                             format(multiplier, adder, one_based, fit,
                                    plaintext[:50]))
                if fit > best_fit:
                    best_fit = fit
                    best_multiplier = multiplier
                    best_adder = adder
                    best_one_based = one_based
    logger.info('Affine break best fit with key {0}x+{1} ({2}) gives fit of '
                '{3} and decrypt starting: {4}'.format(
                    best_multiplier, best_adder, best_one_based, best_fit,
                    affine_decipher(sanitised_message, best_multiplier,
                                    best_adder, best_one_based)[:50]))
    return (best_multiplier, best_adder, best_one_based), best_fit

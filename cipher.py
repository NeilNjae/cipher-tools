import string
import collections


def caesar_cipher_letter(letter, shift):
    if letter in string.ascii_letters:
        if letter in string.ascii_lowercase:
            return chr((ord(letter) - ord('a') + shift) % 26 + ord('a'))
        else:
            new_letter = letter.lower()
            yolo = chr((ord(new_letter) - ord('a') + shift) % 26 + ord('a'))
            return yolo.upper()
    else:
        return letter

def caesar_decipher_letter(letter, shift):
    return caesar_cipher_letter(letter, -shift)

def caesar_cipher_message(message, shift):
    big_cipher = [caesar_cipher_letter(l, shift) for l in message]
    return ''.join(big_cipher)

def caesar_decipher_message(message, shift):
    return caesar_cipher_message(message, -shift)

def letter_frequencies(message):
    frequencies = collections.defaultdict(int)
    for letter in message: 
        if letter in  string.ascii_letters:
            frequencies[letter.lower()]+=1
    return frequencies


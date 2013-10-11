import string
import collections

english_counts = collections.defaultdict(int)
with open('count_1l.txt', 'r') as f:
    for line in f:
        (letter, count) = line.split("\t")
        english_counts[letter] = int(count)

def sanitise(text):
    sanitised = [c.lower() for c in text if c in string.ascii_letters]
    return ''.join(sanitised)

def letter_frequencies(message):
    frequencies = collections.defaultdict(int)
    for letter in sanitise(message): 
        frequencies[letter]+=1
    return frequencies

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

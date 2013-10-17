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

def scale_freq(frequencies):
    total= sum(frequencies.values())
    scaled_frequencies = collections.defaultdict(int)
    for letter in frequencies.keys():
        scaled_frequencies[letter] = frequencies[letter] / total
    return scaled_frequencies

def value_diff(frequencies1, frequencies2):
    total= 0
    for letter in frequencies1.keys():
        total += abs(frequencies1[letter]-frequencies2[letter])
    return total
        
    

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

def caesar_break(message):
    best_key = 0
    best_fit = float("inf")
    for shift in range(26):
        plaintxt = caesar_decipher_message(message, shift)
        lettertxt = letter_frequencies(plaintxt)
        total1 = scale_freq(lettertxt)
        total2 = scale_freq(english_counts)
        fit = value_diff(total2, total1)
        if fit < best_fit:
            best_key = shift
            best_fit = fit
    return best_key

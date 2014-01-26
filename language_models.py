import norms
import itertools
import random
import bisect
import collections

english_counts = collections.defaultdict(int)
with open('count_1l.txt', 'r') as f:
    for line in f:
        (letter, count) = line.split("\t")
        english_counts[letter] = int(count)
normalised_english_counts = norms.normalise(english_counts)

english_bigram_counts = collections.defaultdict(int)
with open('count_2l.txt', 'r') as f:
    for line in f:
        (bigram, count) = line.split("\t")
        english_bigram_counts[bigram] = int(count)
normalised_english_bigram_counts = norms.normalise(english_bigram_counts)

english_trigram_counts = collections.defaultdict(int)
with open('count_3l.txt', 'r') as f:
    for line in f:
        (trigram, count) = line.split("\t")
        english_trigram_counts[trigram] = int(count)
normalised_english_trigram_counts = norms.normalise(english_trigram_counts)

with open('words.txt', 'r') as f:
    keywords = [line.rstrip() for line in f]

def weighted_choice(d):
	"""Generate a set of random items from a dictionary of item counts
	"""
	target = random.uniform(0, sum(d.values()))
	cuml = 0.0
	for (l, p) in d.items():
		cuml += p
		if cuml > target:
			return l
	return None

def random_english_letter():
	"""Generate a random letter based on English letter counts
	"""
	return weighted_choice(normalised_english_counts)

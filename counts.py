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


# choices, weights = zip(*weighted_choices)
# cumdist = list(itertools.accumulate(weights))
# x = random.random() * cumdist[-1]
# choices[bisect.bisect(cumdist, x)]

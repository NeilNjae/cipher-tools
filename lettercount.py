import collections
import string

def sanitise(text):
    return [l.lower() for l in text if l in string.ascii_letters]

corpora = ['shakespeare.txt', 'sherlock-holmes.txt', 'war-and-peace.txt']
counts = collections.defaultdict(int)

for corpus in corpora:
    text = sanitise(open(corpus, 'r').read())
    for letter in text:
        counts[letter] += 1

sorted_letters = sorted(counts, key=counts.get, reverse=True)

with open('count_1l.txt', 'w') as f:
    for l in sorted_letters:
        f.write("{0}\t{1}\n".format(l, counts[l]))
        
    
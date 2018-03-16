import collections
import string
from utilities import sanitise

corpora = ['shakespeare.txt', 'sherlock-holmes.txt', 'war-and-peace.txt']
counts = collections.Counter()

for corpus in corpora:
    text = sanitise(open(corpus, 'r').read())
    counts.update(text)

sorted_letters = sorted(counts, key=counts.get, reverse=True)

with open('count_1l.txt', 'w') as f:
    for l in sorted_letters:
        f.write("{0}\t{1}\n".format(l, counts[l]))
        
    
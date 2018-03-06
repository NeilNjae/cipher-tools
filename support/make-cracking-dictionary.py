import language_models

american = set(open('/usr/share/dict/american-english', 'r').readlines())
british = set(open('/usr/share/dict/british-english', 'r').readlines())
cracklib = set(open('/usr/share/dict/cracklib-small', 'r').readlines())

words = american | british | cracklib

# sanitised_words = set()

# for w in words:
    # sanitised_words.add(language_models.sanitise(w))
    
sanitised_words = set(language_models.sanitise(w) for w in words)

sanitised_words.discard('')

with open('words.txt', 'w') as f:
    f.write('\n'.join(sorted(sanitised_words, key=lambda w: (len(w), w))))

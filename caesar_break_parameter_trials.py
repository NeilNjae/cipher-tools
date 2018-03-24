import random
import csv
from support.utilities import *
from support.language_models import *
from support.norms import *
from cipher.caesar import *

trials = 100

corpus = sanitise(cat([
    open('support/shakespeare.txt').read(), 
    open('support/sherlock-holmes.txt').read(), 
    open('support/war-and-peace.txt').read()
    ]))
corpus_length = len(corpus)

euclidean_scaled_english_counts = euclidean_scale(english_counts)

metrics = [{'func': l1, 'invert': True, 'name': 'l1'}, 
    {'func': l2, 'invert': True, 'name': 'l2'},
    {'func': l3, 'invert': True, 'name': 'l3'},
    {'func': cosine_similarity, 'invert': False, 'name': 'cosine_similarity'}]

scalings = [{'corpus_frequency': normalised_english_counts, 
         'scaling': normalise,
         'name': 'normalised'},
        {'corpus_frequency': euclidean_scaled_english_counts, 
         'scaling': euclidean_scale,
         'name': 'euclidean_scaled'}]

message_lengths = [100, 50, 30, 20, 10, 5]

def make_frequency_compare_function(
        target_frequency, frequency_scaling, metric, invert):
    def frequency_compare(text):
        counts = frequency_scaling(frequencies(text))
        if invert:
            score = -1 * metric(target_frequency, counts)
        else:
            score = metric(target_frequency, counts)
        return score
    return frequency_compare

models = (
    [ {'func': make_frequency_compare_function(
            s['corpus_frequency'], s['scaling'], 
            m['func'], m['invert']),
       'name': '{} + {}'.format(m['name'], s['name'])}
        for m in metrics
        for s in scalings ] 
    + 
    [{'func': Pletters, 'name': 'Pletters'}, 
     {'func': Pbigrams, 'name': 'Pbigrams'},
     {'func': Ptrigrams, 'name': 'Ptrigrams'}]
)

def random_ciphertext(message_length):
    sample_start = random.randint(0, corpus_length - message_length)
    sample = corpus[sample_start:(sample_start + message_length)]
    key = random.randint(1, 25)
    ciphertext = caesar_encipher(sample, key)
    return key, ciphertext


def eval_models():
    return {m['name']: {l: eval_one_model(m, l) for l in message_lengths}
               for m in models}

def eval_one_model(model, message_length):
    print(model['name'], message_length)
    successes = 0
    for _ in range(trials):
        key, ciphertext = random_ciphertext(message_length)
        found_key, _ = caesar_break(ciphertext, model['func'])
        if found_key == key:
            successes += 1 
    return successes

def write_results(scores):
    with open('caesar_break_parameter_trials.csv', 'w') as f:
        writer = csv.DictWriter(f, ['name'] + message_lengths, 
            quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        for scoring in sorted(scores):
            scores[scoring]['name'] = scoring
            writer.writerow(scores[scoring])

scores = eval_models()
write_results(scores)

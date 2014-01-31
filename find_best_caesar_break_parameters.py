import random
import collections
from cipher import *
from cipherbreak import *
import itertools

corpus = sanitise(''.join([open('shakespeare.txt', 'r').read(), 
    open('sherlock-holmes.txt', 'r').read(), 
    open('war-and-peace.txt', 'r').read()]))
corpus_length = len(corpus)

euclidean_scaled_english_counts = norms.euclidean_scale(english_counts)

metrics = [{'func': norms.l1, 'name': 'l1'}, 
    {'func': norms.l2, 'name': 'l2'},
    {'func': norms.l3, 'name': 'l3'},
    {'func': norms.cosine_distance, 'name': 'cosine_distance'},
    {'func': norms.harmonic_mean, 'name': 'harmonic_mean'},
    {'func': norms.geometric_mean, 'name': 'geometric_mean'},
    {'func': norms.inverse_log_pl, 'name': 'inverse_log_pl'}]
scalings = [{'corpus_frequency': normalised_english_counts, 
         'scaling': norms.normalise,
         'name': 'normalised'},
        {'corpus_frequency': euclidean_scaled_english_counts, 
         'scaling': norms.euclidean_scale,
         'name': 'euclidean_scaled'},
         {'corpus_frequency': normalised_english_counts,
         'scaling': norms.identity_scale,
         'name': 'normalised_with_identity'}]
message_lengths = [300, 100, 50, 30, 20, 10, 5]

trials = 5000

scores = collections.defaultdict(int)

def eval_all():
    list(itertools.starmap(eval_one_parameter_set,
        itertools.product(metrics, scalings, message_lengths)))

def eval_one_parameter_set(metric, scaling, message_length):
    for _ in range(trials):
        sample_start = random.randint(0, corpus_length - message_length)
        sample = corpus[sample_start:(sample_start + message_length)]
        key = random.randint(1, 25)
        sample_ciphertext = caesar_encipher(sample, key)
        found_key, _ = caesar_break(sample_ciphertext, 
                                          metric=metric['func'], 
                                          target_counts=scaling['corpus_frequency'], 
                                          message_frequency_scaling=scaling['scaling'])
        if found_key == key:
            scores[(metric['name'], scaling['name'], message_length)] += 1 
    return scores[(metric['name'], scaling['name'], message_length)]

def show_results():
    with open('caesar_break_parameter_trials.csv', 'w') as f:
        print(',message_length', file = f)
        print('metric+scaling,', ','.join([str(l) for l in message_lengths]), file = f)
        for (metric, scaling) in itertools.product(metrics, scalings):
            print('{}:{}'.format(metric['name'], scaling['name']), end='', file=f)
            for l in message_lengths:
                print(',', scores[(metric['name'], scaling['name'], l)] / trials, end='', file=f)
            print('', file = f)

eval_all()
show_results()

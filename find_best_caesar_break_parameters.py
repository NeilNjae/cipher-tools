import random
from cipher import *


corpus = sanitise(''.join([open('shakespeare.txt', 'r').read(), open('sherlock-holmes.txt', 'r').read(), open('war-and-peace.txt', 'r').read()]))
corpus_length = len(corpus)

scaled_english_counts = norms.scale(english_counts)


metrics = [norms.l1, norms.l2, norms.l3, norms.cosine_distance, norms.harmonic_mean, norms.geometric_mean]
corpus_frequencies = [normalised_english_counts, scaled_english_counts]
scalings = [norms.normalise, norms.scale]
message_lengths = [300, 100, 50, 30, 20, 10, 5]

metric_names = ['l1', 'l2', 'l3', 'cosine_distance', 'harmonic_mean', 'geometric_mean']
corpus_frequency_names = ['normalised_english_counts', 'scaled_english_counts']
scaling_names = ['normalise', 'scale']

trials = 5000

scores = collections.defaultdict(int)
for metric in range(len(metrics)):
    scores[metric_names[metric]] = collections.defaultdict(int)
    for corpus_freqency in range(len(corpus_frequencies)):
        scores[metric_names[metric]][corpus_frequency_names[corpus_freqency]] = collections.defaultdict(int)
        for scaling in range(len(scalings)):
            scores[metric_names[metric]][corpus_frequency_names[corpus_freqency]][scaling_names[scaling]] = collections.defaultdict(int)
            for message_length in message_lengths:
                for i in range(trials):
                    sample_start = random.randint(0, corpus_length - message_length)
                    sample = corpus[sample_start:(sample_start + message_length)]
                    key = random.randint(1, 25)
                    sample_ciphertext = caesar_encipher(sample, key)
                    (found_key, score) = caesar_break(sample_ciphertext, 
                                                      metric=metrics[metric], 
                                                      target_frequencies=corpus_frequencies[corpus_freqency], 
                                                      message_frequency_scaling=scalings[scaling])
                    if found_key == key:
                        scores[metric_names[metric]][corpus_frequency_names[corpus_freqency]][scaling_names[scaling]][message_length] += 1 
                print(', '.join([metric_names[metric], 
                                 corpus_frequency_names[corpus_freqency], 
                                 scaling_names[scaling], 
                                 str(message_length), 
                                 str(scores[metric_names[metric]][corpus_frequency_names[corpus_freqency]][scaling_names[scaling]][message_length] / trials) ]))


with open('caesar_break_parameter_trials.csv', 'w') as f:
    for metric in range(len(metrics)):
        for corpus_freqency in range(len(corpus_frequencies)):
            for scaling in range(len(scalings)):
                for message_length in message_lengths:
                    print(', '.join([metric_names[metric], 
                                     corpus_frequency_names[corpus_freqency], 
                                     scaling_names[scaling], 
                                     str(message_length), 
                                     str(scores[metric_names[metric]][corpus_frequency_names[corpus_freqency]][scaling_names[scaling]][message_length] / trials) ]), 
                          file=f)
                      
                            
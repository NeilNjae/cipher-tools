import matplotlib.pyplot as plt

def plot_frequency_histogram(freqs, sort_key=None):
    x = range(len(freqs))
    y = [freqs[l] for l in sorted(freqs, key=sort_key)]
    f = plt.figure()
    ax = f.add_axes([0.1, 0.1, 0.9, 0.9])
    ax.bar(x, y, align='center')
    ax.set_xticks(x)
    ax.set_xticklabels(sorted(freqs, key=sort_key))
    f.show()

if __name__ == "__main__":
    import doctest
    doctest.testmod()

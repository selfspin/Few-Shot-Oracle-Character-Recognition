import numpy as np
import random
import matplotlib.pyplot as plt

npz_path = '../Oracle/data/oracle_fs/seq/oracle_200_1_shot/194.npz'
data = np.load(npz_path, encoding='latin1', allow_pickle=True)
train_data = data['train']
test_data = data['test']


sample = train_data[0]


def strokes_to_lines(strokes):
    """Convert stroke-3 format to polyline format."""
    x = 0
    y = 0
    lines = []
    line = []

    for i in range(len(strokes)):
        if strokes[i, 2] == 1:
            x += float(strokes[i, 0])
            y += float(strokes[i, 1])
            line.append([x, y])
            lines.append(line)
            line = []
        else:
            x += float(strokes[i, 0])
            y += float(strokes[i, 1])
            line.append([x, y])
    return lines


def show_one_sample(strokes, linewidth=10):
    lines = strokes_to_lines(strokes)
    for idx in range(0, len(lines)):
        x = [x[0] for x in lines[idx]]
        y = [y[1] for y in lines[idx]]
        plt.plot(x, y, 'k-', linewidth=linewidth)

    ax = plt.gca()
    plt.xticks([])
    plt.yticks([])
    ax.xaxis.set_ticks_position('top')
    ax.invert_yaxis()

    plt.show()


# show
show_one_sample(sample)
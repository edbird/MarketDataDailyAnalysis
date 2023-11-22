#!/usr/bin/env python3


import numpy
import scipy.stats as stats

import matplotlib.pyplot as plt


def main():

    x = numpy.linspace(-5.0, 5.0, 100)
    y1 = stats.norm.pdf(x, 0.0, 1.0)
    y2 = stats.norm.pdf(x, 1.0, 0.5)
    y3 = stats.norm.pdf(x, -0.5, 2.0)

    figure = plt.figure()

    ax = figure.add_subplot(1, 1, 1)

    ax.plot(x, y1)
    ax.plot(x, y2)
    ax.plot(x, y3)

    figure.savefig('test.png')


if __name__ == '__main__':
    main()


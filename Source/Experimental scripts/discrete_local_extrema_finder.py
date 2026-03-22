import numpy as np
from scipy.signal import argrelextrema


def local_extrema(array, order):
    """
    Finds the indices of both the local maxima and minima in a 1D array dataset.
    :param array: 1D array data
    :param order: How many points on each side to use for the comparison to consider to be true
    :return: arrays of the maxima and minima indices from the input 1D array
    """
    # for local maxima
    maxima = argrelextrema(array, np.greater, order=order)

    # for local minima
    minima = argrelextrema(array, np.less, order=order)

    return maxima, minima


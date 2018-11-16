# https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python

import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

import numpy as np

def biased_flip(p, shape):
    temp = np.random.random(shape)
    return np.where(temp <= p, 1, 0)

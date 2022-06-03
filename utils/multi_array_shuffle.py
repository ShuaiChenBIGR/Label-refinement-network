import numpy as np

def shuffle_mdarray(obj):
    """
    Assumes that obj is a numpy array
    """
    shape = obj.shape
    obj = obj.ravel()
    np.random.shuffle(obj)
    return obj.reshape(shape)
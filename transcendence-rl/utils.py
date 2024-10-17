import numpy as np

def flatten_dict_concat(d):
    # Flatten arrays and convert scalars to arrays before concatenating
    return np.concatenate([np.atleast_1d(v).flatten() for v in d.values()])
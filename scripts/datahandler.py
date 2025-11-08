import numpy as np
import itertools
import random

def RandomSample(array,
                 n: int | None = None,
                 axis: int = 0,
                 seed: list | np.ndarray = [],
                 low: int | None = None,
                 high: int | None = None,
                 random_state=42,
                 return_seed: bool = False):
    '''
    Recieves an np.array and selects a random subset to return as a sample.
    axis:           dimension to sample from, singular.
    seed:           a preset selection (list)
    min:            start for selection if no seed
    max:            end of selection if no seed
    random_state:   used for random selection if not seed
    return_seed:    whether or not to return the boolean selection list used
    '''
    low = low if low else 0
    high = high if high else array.shape[axis]
    n = n if n else int(array.shape[axis]/2)
    seed = list(seed) if any(seed) else []

    if any(seed):
        sample = list(itertools.compress(array, seed))
    else:
        options = [i for i in range(low, high)]
        rawseed = random.sample(options, n)
        seed = [0 for i in range(0,low)]
        for option in options:
            if option in rawseed:
                seed.append(1)
            else:
                seed.append(0)
        for i in range(high, array.shape[axis]):
            seed.append(0)
        sample = list(itertools.compress(array, seed))
    
        seed = np.asarray(seed)

    if return_seed:
        return sample, seed
    else:
        return sample
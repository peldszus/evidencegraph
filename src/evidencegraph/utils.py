"""
Created on 20.05.2016

@author: Andreas Peldszus
"""

from itertools import islice
from hashlib import md5


def window(seq, n=2):
    """
    Returns a sliding window (of width n) over data from the iterable
    (from itertools examples)

    >>> list(window([1, 2, 3, 4], n=2))
    [(1, 2), (2, 3), (3, 4)]
    >>> list(window([1, 2, 3, 4], n=3))
    [(1, 2, 3), (2, 3, 4)]
    >>> list(window([1, 2], n=3))
    []
    """
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def split(a, n):
    """
    http://stackoverflow.com/a/2135920
    """
    k, m = len(a) // n, len(a) % n
    return (
        a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)
    )


def foldsof(X, y, n=3):
    """
    A simple folding of X,y data, splitting linearily.

    >>> X = [0,1,2,3,4,5,6,7,8,9]
    >>> y = list('abcaabacbc')
    >>> list(foldsof(X, y, n=3))
    [(((4, 5, 6, 7, 8, 9), ('a', 'b', 'a', 'c', 'b', 'c')),
      ((0, 1, 2, 3), ('a', 'b', 'c', 'a'))),
     (((0, 1, 2, 3, 7, 8, 9), ('a', 'b', 'c', 'a', 'c', 'b', 'c')),
      ((4, 5, 6), ('a', 'b', 'a'))),
     (((0, 1, 2, 3, 4, 5, 6), ('a', 'b', 'c', 'a', 'a', 'b', 'a')),
      ((7, 8, 9), ('c', 'b', 'c')))]

    """
    assert len(X) == len(y)
    assert len(X) >= n
    splits = list(split(list(zip(X, y)), n))
    for n, nth_split in enumerate(splits):
        test_X, test_y = zip(*nth_split)
        train = [e for i, l in enumerate(splits) if i != n for e in l]
        train_X, train_y = zip(*train)
        yield (train_X, train_y), (test_X, test_y)


def hash_of_featureset(features):
    """
    Returns a short hash id of a list of feature names.

    >>> features = ['default', 'bow', 'bow_2gram']
    >>> hash_of_featureset(features)
    '4518ca2'
    """
    return md5(" ".join(sorted(features)).encode()).hexdigest()[:7]

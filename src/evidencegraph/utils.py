# -*- mode: python; coding: utf-8; -*-

'''
Created on 20.05.2016

@author: Andreas Peldszus
'''

import os
from itertools import islice
from hashlib import md5
import joblib

from arggraph import ArgGraph
from argtree import ArgTree
from argtree import RELATION_SETS
from argtree import SIMPLE_RELATION_SET


CORPUS_PATH_DE_ADU = 'data/corpus/german/arg/'
CORPUS_PATH_EN_ADU = 'data/corpus/english/arg/'
CORPUS_PATH_EN_EDU = 'data/corpus/english/arg_fine/'

CORPUS_PATH_BY_ID = {
    'm3-v1': 'data/corpus/english/m3-v1/'
}


# deprecated
def load_graph_corpus(path, silent=True):
    r = {}
    _, _, filenames = os.walk(path).next()
    for i in filenames:
        if i.endswith('.xml'):
            if not silent:
                print i, '...'
            g = ArgGraph()
            g.load_from_xml(path + i)
            r[g.graph['id']] = g
    return r


# deprecated
def trees_from_graphs(graph_corpus, segmentation, relation_set):
    from_adu = segmentation == 'adu'
    long_names = relation_set != SIMPLE_RELATION_SET
    tree_corpus = {}
    for id_, arggraph in graph_corpus.iteritems():
        tree = ArgTree(relation_set=relation_set)
        tree.load_from_arggraph(arggraph, from_adus=from_adu,
                                long_names=long_names)
        tree_corpus[id_] = tree
    return tree_corpus


# deprecated
def segments_from_graphs(graph_corpus, segmentation):
    texts = {}
    for id_, arggraph in graph_corpus.iteritems():
        if segmentation == 'adu':
            texts[id_] = arggraph.get_adu_segmented_text()
        else:
            texts[id_] = arggraph.get_segmented_text()
    return texts


# deprecated
def load_corpus(language, segmentation, relationset, corpus_id=None):
    assert language in ['de', 'en']
    assert segmentation in ['adu', 'edu']
    assert relationset in RELATION_SETS
    if corpus_id:
        try:
            corpus_path = CORPUS_PATH_BY_ID[corpus_id]
        except KeyError:
            assert False, "Corpus id not known."
    elif language == 'de':
        if segmentation == 'adu':
            corpus_path = CORPUS_PATH_DE_ADU
        else:
            assert False, "Only the English corpus is EDU segmented."
    else:
        if segmentation == 'adu':
            corpus_path = CORPUS_PATH_EN_ADU
        else:
            corpus_path = CORPUS_PATH_EN_EDU
    graph_corpus = load_graph_corpus(corpus_path)
    trees = trees_from_graphs(graph_corpus, segmentation, relationset)
    texts = segments_from_graphs(graph_corpus, segmentation)
    return texts, trees


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
    k, m = len(a) / n, len(a) % n
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]
            for i in xrange(n))


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
    splits = list(split(zip(X, y), n))
    for n in range(len(splits)):
        test_X, test_y = zip(*splits[n])
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
    return md5(' '.join(sorted(features))).hexdigest()[:7]

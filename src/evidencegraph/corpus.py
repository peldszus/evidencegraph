# -*- mode: python; coding: utf-8; -*-

'''
Created on 18.09.2017

@author: Andreas Peldszus
'''

import os

from arggraph import ArgGraph
from argtree import ArgTree
from argtree import RELATION_SETS
from argtree import SIMPLE_RELATION_SET
from folding import RepeatedGroupwiseStratifiedKFold


CORPORA = {
    'm112de': {
        'language': 'de',
        'path': 'data/corpus/german/arg/'
    },
    'm112en': {
        'language': 'en',
        'path': 'data/corpus/english/arg/'
    },
    'm112en_fine': {
        'language': 'en',
        'path': 'data/corpus/english/arg_fine/'
    },
    'm3-v5': {
        'language': 'en',
        'path': 'data/corpus/english/m3-v5/'
    }
}


class GraphCorpus(object):

    def __init__(self):
        self.graphs = {}

    def load(self, path, silent=True):
        """
        Loads the graphs of the corpus and return the freshly
        added text ids.
        """
        _, _, filenames = os.walk(path).next()
        ids = []
        for fn in filenames:
            if fn.endswith('.xml'):
                if not silent:
                    print fn, '...'
                g = ArgGraph()
                g.load_from_xml(os.path.join(path, fn))
                graph_id = g.graph['id']
                assert graph_id not in self.graphs
                # test integrity
                try:
                    if g.get_role_type_labels().values():
                        self.graphs[graph_id] = g
                        ids.append(graph_id)
                except Exception as e:
                    print "Could not load {} :".format(fn), e
        return ids

    def segments(self, segmentation):
        """
        Returns all texts of the corpus in 'adu' or 'edu'
        segmentation, in the form {text_id: [segment, ...]}
        """
        texts = {}
        if segmentation == 'adu':
            segment_getter = ArgGraph.get_adu_segmented_text_with_restatements
        else:
            segment_getter = ArgGraph.get_segmented_text

        for id_, graph in self.graphs.iteritems():
            texts[id_] = segment_getter(graph)
        return texts

    def trees(self, segmentation, relation_set):
        """
        Returns all dependency trees converted from the
        graphs of the corpus in 'adu' or 'edu' segmentation
        for relation_set, in the form {text_id: tree}.
        """
        from_adu = segmentation == 'adu'
        long_names = relation_set != SIMPLE_RELATION_SET
        tree_corpus = {}
        for id_, arggraph in self.graphs.iteritems():
            tree = ArgTree(relation_set=relation_set)
            tree.load_from_arggraph(
                arggraph, from_adus=from_adu, long_names=long_names)
            tree_corpus[id_] = tree
        return tree_corpus

    def role_type_labels(self):
        """
        TBA
        """
        # return {
        #     tid: graph.get_role_type_labels().values()
        #     for tid, graph in self.graphs.iteritems()
        # }
        labels = {}
        for tid, graph in self.graphs.iteritems():
            try:
                labels[tid] = graph.get_role_type_labels().values()
            except Exception:
                print "Error", tid
                pass
        return labels

    def create_folds(self, number=5, shuffle=True, seed=1, repeats=10):
        """
        TBA
        """
        return RepeatedGroupwiseStratifiedKFold(
            number, self.role_type_labels(),
            shuffle=shuffle, seed=seed, repeats=repeats
        )

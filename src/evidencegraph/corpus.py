"""
Created on 18.09.2017

@author: Andreas Peldszus
"""


import os

from .arggraph import ArgGraph
from .argtree import ArgTree
from .argtree import SIMPLE_RELATION_SET
from .folding import RepeatedGroupwiseStratifiedKFold


CORPORA = {
    "m112de": {
        "language": "de",
        "path": "data/corpus/arg-microtexts-master/corpus/de/",
    },
    "m112en": {
        "language": "en",
        "path": "data/corpus/arg-microtexts-master/corpus/en/",
    },
    "m112en_fine": {
        "language": "en",
        "path": "data/corpus/arg-microtexts-multilayer-master/corpus/arg/",
    },
    "m112en_part2": {
        "language": "en",
        "path": "data/corpus/arg-microtexts-part2-master/corpus/",
    },
}


class GraphCorpus(object):
    def __init__(self):
        self.graphs = {}

    def load(self, path, silent=True):
        """
        Loads the graphs of the corpus and return the freshly
        added text ids.
        """
        _, _, filenames = next(os.walk(path))
        ids = []
        for fn in filenames:
            if fn.endswith(".xml"):
                if not silent:
                    print(fn, "...")
                g = ArgGraph()
                g.load_from_xml(os.path.join(path, fn))
                graph_id = g.graph["id"]
                assert graph_id not in self.graphs
                # test integrity
                try:
                    if g.get_role_type_labels().values():
                        self.graphs[graph_id] = g
                        ids.append(graph_id)
                except Exception as e:
                    print("Could not load {} :".format(fn), e)
        return ids

    def segments(self, segmentation):
        """
        Returns all texts of the corpus in 'adu' or 'edu'
        segmentation, in the form {text_id: [segment, ...]}
        """
        texts = {}
        if segmentation == "adu":
            segment_getter = ArgGraph.get_adu_segmented_text_with_restatements
        else:
            segment_getter = ArgGraph.get_segmented_text

        for id_, graph in self.graphs.items():
            texts[id_] = segment_getter(graph)
        return texts

    def trees(self, segmentation, relation_set):
        """
        Returns all dependency trees converted from the
        graphs of the corpus in 'adu' or 'edu' segmentation
        for relation_set, in the form {text_id: tree}.
        """
        from_adu = segmentation == "adu"
        long_names = relation_set != SIMPLE_RELATION_SET
        tree_corpus = {}
        for id_, arggraph in self.graphs.items():
            tree = ArgTree(relation_set=relation_set)
            tree.load_from_arggraph(
                arggraph, from_adus=from_adu, long_names=long_names
            )
            tree_corpus[id_] = tree
        return tree_corpus

    def segments_trees(self, segmentation, relation_set):
        """Returns a tuple of both segments and trees."""
        return (
            self.segments(segmentation),
            self.trees(segmentation, relation_set),
        )

    def role_type_labels(self):
        """
        TBA
        """
        # return {
        #     tid: graph.get_role_type_labels().values()
        #     for tid, graph in self.graphs.iteritems()
        # }
        labels = {}
        for tid, graph in self.graphs.items():
            try:
                labels[tid] = list(graph.get_role_type_labels().values())
            except Exception:
                print("Error", tid)
                pass
        return labels

    def create_folds(self, number=5, shuffle=True, seed=1, repeats=10):
        """
        TBA
        """
        return RepeatedGroupwiseStratifiedKFold(
            number,
            self.role_type_labels(),
            shuffle=shuffle,
            seed=seed,
            repeats=repeats,
        )


def combine_corpora(corpora, mode="normal"):
    """
    Combine multiple corpora into one corpus and return the corpus
    and the folds.

    corpora: list of corpus ids to load
    mode:
        'normal' = train and test folded on all corpora
                   A" + B" + C" >> A' + B' + C'
        'cross'  = train on all but last corpus, test folded on last
                   A + B >> C'
        'add'    = train on all and last corpus, test folded on last
                   A + B + C" >> C'
    """
    assert all(corpus in CORPORA for corpus in corpora)

    if mode == "normal":
        gc = GraphCorpus()
        for corpus in corpora:
            gc.load(CORPORA[corpus]["path"])
        folds = list(gc.create_folds())

    elif mode == "cross":
        gc = GraphCorpus()
        assert len(corpora) > 1
        last = corpora[-1]
        gc.load(CORPORA[last]["path"])
        last_folds = list(gc.create_folds())
        first_corpora_text_ids = []
        for corpus in corpora[:-1]:
            ids = gc.load(CORPORA[corpus]["path"])
            first_corpora_text_ids.extend(ids)
        folds = [
            (first_corpora_text_ids, test, n) for _, test, n in last_folds
        ]

    elif mode == "add":
        gc = GraphCorpus()
        assert len(corpora) > 1
        last = corpora[-1]
        all_corpora_text_ids = []
        ids = gc.load(CORPORA[last]["path"])
        all_corpora_text_ids.extend(ids)
        last_folds = list(gc.create_folds())
        for corpus in corpora[:-1]:
            ids = gc.load(CORPORA[corpus]["path"])
            all_corpora_text_ids.extend(ids)
        folds = [
            ([i for i in all_corpora_text_ids if i not in test], test, n)
            for _, test, n in last_folds
        ]

    return gc, folds

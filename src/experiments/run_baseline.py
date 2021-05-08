"""
@author: Andreas Peldszus
"""


import json
from collections import defaultdict, Counter
from datetime import datetime

from evidencegraph.corpus import GraphCorpus, CORPORA
from evidencegraph.folds import get_static_folds
from evidencegraph.argtree import ArgTree
from evidencegraph.argtree import RELATION_SETS_BY_NAME


class BaselineAttachFirst(object):
    """Pseudo classifier for producing the attach-to-first baseline."""

    def predict(self, number_of_nodes, label):
        """
        Returns an ArgTree with `number_of_nodes`, where every node
        except the first one is connected to the first node with a
        relation of type `label`.
        >>> clf = BaselineAttachFirst()
        >>> tree = clf.predict(4, 'sup')
        >>> tree.get_triples()
        [[2, 1, 'sup'], [3, 1, 'sup'], [4, 1, 'sup']]
        """
        triples = [(i, 1, label) for i in range(2, number_of_nodes + 1)]
        return ArgTree(from_triples=triples)


class BaselineAttachPreceeding(object):
    """Pseudo classifier for producing the attach-to-preceeding baseline."""

    def predict(self, number_of_nodes, label):
        """
        Returns an ArgTree with `number_of_nodes`, where every node
        is connected to its preceeding node with a relation of type `label`.
        >>> clf = BaselineAttachPreceeding()
        >>> tree = clf.predict(4, 'sup')
        >>> tree.get_triples()
        [[2, 1, 'sup'], [3, 2, 'sup'], [4, 3, 'sup']]
        """
        triples = [(i, i - 1, label) for i in range(2, number_of_nodes + 1)]
        return ArgTree(from_triples=triples)


def folds_static(trees, relation_set, baseline):
    # determine the majority relation label
    relation_labels = Counter(
        [
            relation_label
            for tree in trees.values()
            for _, _, relation_label in tree.get_triples()
        ]
    )
    majority_label = relation_labels.most_common(1)[0][0]
    print("Determined '{}' as the majority label.".format(majority_label))

    # produce the predictions
    predictions = defaultdict(dict)
    for _train_tids, test_tids, i in get_static_folds():
        print("[{}] Iteration: {}\t".format(datetime.now(), i))
        if baseline == "first":
            clf = BaselineAttachFirst()
        elif baseline == "prec":
            clf = BaselineAttachPreceeding()
        else:
            raise ValueError("Unknown baseline type.")
        for tid in test_tids:
            gold_length = len(trees[tid].nodes())
            tree = clf.predict(gold_length, majority_label)
            tree.relation_set = relation_set
            predictions[i][tid] = tree.get_triples()

    return predictions


if __name__ == "__main__":
    conditions = {
        # de adu
        "m112de-diss-adu-simple-baseline-first": {
            "corpus": "m112de",
            "relation_set": "SIMPLE_RELATION_SET",
            "segmentation": "adu",
            "baseline": "first",
        },
        "m112de-diss-adu-simple-baseline-prec": {
            "corpus": "m112de",
            "relation_set": "SIMPLE_RELATION_SET",
            "segmentation": "adu",
            "baseline": "prec",
        },
        "m112de-diss-adu-full-baseline-first": {
            "corpus": "m112de",
            "relation_set": "FULL_RELATION_SET_ADU",
            "segmentation": "adu",
            "baseline": "first",
        },
        "m112de-diss-adu-full-baseline-prec": {
            "corpus": "m112de",
            "relation_set": "FULL_RELATION_SET_ADU",
            "segmentation": "adu",
            "baseline": "prec",
        },
        # en adu
        "m112en-diss-adu-simple-baseline-first": {
            "corpus": "m112en",
            "relation_set": "SIMPLE_RELATION_SET",
            "segmentation": "adu",
            "baseline": "first",
        },
        "m112en-diss-adu-simple-baseline-prec": {
            "corpus": "m112en",
            "relation_set": "SIMPLE_RELATION_SET",
            "segmentation": "adu",
            "baseline": "prec",
        },
        "m112en-diss-adu-full-baseline-first": {
            "corpus": "m112en",
            "relation_set": "FULL_RELATION_SET_ADU",
            "segmentation": "adu",
            "baseline": "first",
        },
        "m112en-diss-adu-full-baseline-prec": {
            "corpus": "m112en",
            "relation_set": "FULL_RELATION_SET_ADU",
            "segmentation": "adu",
            "baseline": "prec",
        },
        # en edu
        "m112en-diss-edu-simple-baseline-first": {
            "corpus": "m112en_fine",
            "relation_set": "SIMPLE_RELATION_SET",
            "segmentation": "edu",
            "baseline": "first",
        },
        "m112en-diss-edu-simple-baseline-prec": {
            "corpus": "m112en_fine",
            "relation_set": "SIMPLE_RELATION_SET",
            "segmentation": "edu",
            "baseline": "prec",
        },
        "m112en-diss-edu-full-baseline-first": {
            "corpus": "m112en_fine",
            "relation_set": "FULL_RELATION_SET",
            "segmentation": "edu",
            "baseline": "first",
        },
        "m112en-diss-edu-full-baseline-prec": {
            "corpus": "m112en_fine",
            "relation_set": "FULL_RELATION_SET",
            "segmentation": "edu",
            "baseline": "prec",
        },
    }

    # run all experiment conditions
    for condition, params in conditions.items():
        print("### Running experiment condition", condition)
        corpus = GraphCorpus()
        corpus.load(CORPORA[params["corpus"]]["path"])
        relation_set = RELATION_SETS_BY_NAME[params["relation_set"]]
        trees = corpus.trees(params["segmentation"], relation_set)
        predictions = folds_static(trees, relation_set, params["baseline"])
        with open("data/{}.json".format(condition), "w") as f:
            json.dump(predictions, f, indent=1, sort_keys=True)

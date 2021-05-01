# -*- coding: utf-8 -*-

from collections import defaultdict, deque
from itertools import chain
import random

example_data_1 = {
    "g01": "BBDE",
    "g02": "CBEDF",
    "g03": "BBAA",
    "g04": "ABCD",
    "g05": "ABBDF",
    "g06": "ABC",
    "g07": "ABBAA",
    "g08": "ACBD",
    "g09": "DEBBA",
    "g10": "AABC",
    "g11": "AAAAF",
    "g12": "CCDACD",
    "g13": "CCADB",
    "g14": "CBAF",
    "g15": "ABCD",
    "g16": "CDBD",
}

example_data_2 = {
    "g01": "BBDE",
    "g02": "CBEDF",
    "g03": "BBAA",
    "g04": "ABCD",
    "g05": "ABBDF",
    "g06": "ABC",
    "g07": "ABBAA",
    "g08": "ACBD",
    "g09": "DEBBA",
    "g10": "AABC",
    "g11": "AAAAF",
    "g12": "CCDACD",
    "g13": "CCADB",
    "g14": "CBAF",
    "g15": "ABCD",
    "g16": "CDBD",
    "g17": "BBDE",
    "g18": "CBEDF",
    "g19": "BBAA",
    "g20": "ABCD",
    "g21": "ABBDF",
    "g22": "ABC",
    "g23": "ABBAA",
    "g24": "ACBD",
    "g25": "DEBBA",
    "g26": "AABC",
    "g27": "AAAAF",
    "g28": "CCDACD",
    "g29": "CCADB",
    "g30": "CBAF",
    "g31": "ABCD",
    "g32": "CDBD",
}


def absolute_class_counts(data, expected_classes=None):
    """input: a dict mapping a group key to a list of class occurrences
           [0,2,2,1,0,1,2,2,0]
    output: a dict mapping class keys to their absolute counts
           {0:3, 1:2, 2:4}"""
    counts_class = defaultdict(int)
    if expected_classes is not None:
        for c in expected_classes:
            counts_class[c]
    for e in data:
        counts_class[e] += 1
    return counts_class


def relative_class_counts(data):
    """input: a dict mapping class keys to their absolute counts
    output: a dict mapping class keys to their relative counts"""
    counts_items = sum(data.values())
    return {k: 1.0 * v / counts_items for k, v in data.items()}


def diff_distribution(a, b, weights=None):
    """compares two distributions and returns a sum of all (weighted)
    diffs"""
    assert a.keys() == b.keys()
    if weights is not None:
        assert a.keys() == weights.keys()
        diff = {k: weights[k] * abs(a[k] - b[k]) for k in a}
    else:
        diff = {k: abs(a[k] - b[k]) for k in a}
    return sum(diff.values())


def join_distributions(a, b):
    """joins two distributions of absolute class counts by adding the values
    of each key"""
    assert a.keys() == b.keys()
    return {k: a[k] + b[k] for k in a}


class GroupwiseStratifiedKFold(object):
    def __init__(self, number_of_folds, data, shuffle=False, seed=0):
        """
        Groupwise, stratified k-fold splits of a dataset for validation.
        It is stratified, because the label distributions aim to be similar
        across the splits. It is groupwise, because classification items are grouped,
        i.e. considered belonging together, so that not items but groups of items
        are sampled.

        In our case we sample classification items grouped together because they
        belong to one input text, so that the kfold does not contain fragments of texts.

        The input `data` is considered to be a dict mapping from group ids to lists of
        labels.

        >>> folds = list(GroupwiseStratifiedKFold(4, example_data_1))

        Correct length of folds.
        >>> len(folds) == 4
        True

        No overlap between train and tests folds.
        >>> all(set(train) & set(test) == set([]) for train, test in folds)
        True

        Each train/test split covers the whole data.
        >>> all(set(train) | set(test) == set(example_data_1.keys()) for train, test in folds)
        True

        All test sets of the folding cover the whole data.
        >>> set([group for tr, test in folds for group in test]) == set(example_data_1.keys())
        True

        Folding is stratified, i.e. label distributions are similar.
        >>> [''.join(sorted(label for group in train for label in example_data_1[group])) for train, _ in folds]
        ['AAAAAAAAAAAAAABBBBBBBBBBBBBBCCCCCCCCCCDDDDDDDDDDEEFFF',
         'AAAAAAAAAAAAAABBBBBBBBBBBBBBBCCCCCCCCCCCDDDDDDDDDDEEFFF',
         'AAAAAAAAAAAAAAAAABBBBBBBBBBBBBBCCCCCCCCCCDDDDDDDDEEFFF',
         'AAAAAAAAAAAAAAABBBBBBBBBBBBBBCCCCCCCCDDDDDDDDEEEFFF']
        """
        self.fold_register = {}
        ungrouped_data = list(chain(*list(data.values())))
        counts_class_absolute = absolute_class_counts(ungrouped_data)
        counts_class_relative = relative_class_counts(counts_class_absolute)
        classes = list(counts_class_absolute.keys())
        class_weights = {k: 1 - v for k, v in counts_class_relative.items()}
        group_distribution = {
            k: absolute_class_counts(list(v), expected_classes=classes)
            for k, v in data.items()
        }
        folds = {
            n: {k: 0 for k in counts_class_relative}
            for n in range(1, number_of_folds + 1)
        }
        fold_register = {n: [] for n in folds.keys()}
        pool = set(group_distribution.keys())

        cnt_pass = 0
        while len(pool) > 0:
            # either shuffle the order of filling folds in this pass randomly
            # or rotate it, in order to prevent that the first folds or a pass
            # always get the best possible draw from the pool
            if shuffle:
                random.seed(seed + cnt_pass)
                fold_order_in_this_pass = list(folds.keys())
                random.shuffle(fold_order_in_this_pass)
            else:
                fold_order_in_this_pass = deque(folds.keys())
                fold_order_in_this_pass.rotate(-cnt_pass)

            # in a pass, fill each fold with the best group
            for this_fold in fold_order_in_this_pass:
                this_folds_dist = folds[this_fold]
                if len(pool) == 0:
                    break

                # find the group in the pool, that minimizes the difference of
                # this fold to the base distribution
                min_diff = float("+inf")
                min_group = None
                min_joint_dist = None
                for group in pool:
                    joint_dist = join_distributions(
                        this_folds_dist, group_distribution[group]
                    )
                    diff = diff_distribution(
                        counts_class_relative,
                        relative_class_counts(joint_dist),
                        weights=class_weights,
                    )
                    if diff < min_diff:
                        min_diff = diff
                        min_group = group
                        min_joint_dist = joint_dist

                # remove group from pool, register group in fold and add group
                # absolutes to fold
                pool.remove(min_group)
                fold_register[this_fold].append(min_group)
                folds[this_fold] = min_joint_dist

            cnt_pass += 1

        self.fold_register = fold_register

    def __iter__(self):
        """Yields group ids of training and testing items."""
        for test_fold in self.fold_register.keys():
            train_foldes = list(self.fold_register.keys())
            train_foldes.remove(test_fold)
            train_ids_per_fold = [self.fold_register[f] for f in train_foldes]
            train_ids = list(chain(*train_ids_per_fold))
            test_ids = self.fold_register[test_fold]
            yield train_ids, test_ids


class RepeatedGroupwiseStratifiedKFold:
    def __init__(
        self, number_of_folds, data, shuffle=False, seed=0, repeats=10
    ):
        """
        Repeated, groupwise, stratified k-fold splits of a dataset for validation.
        The GroupwiseStratifiedKFold is repeated with different random seeds in order
        to yield different kfold splits of the same dataset.

        >>> folds = list(RepeatedGroupwiseStratifiedKFold(4, example_data_2, shuffle=True, repeats=2))

        Correct length of folds.
        >>> len(folds) == 2 * 4
        True

        Every fold is different. Note this only works when suffle=True and the
        dataset is large enough.
        >>> len(set(tuple(sorted(test)) for _, test, _ in folds)) == 2 * 4
        True
        """
        self.iterations = []
        for repeat_nr in range(repeats):
            foldes = GroupwiseStratifiedKFold(
                number_of_folds, data, shuffle=shuffle, seed=seed + repeat_nr
            )
            for fold_nr, (train, test) in enumerate(foldes):
                self.iterations.append(
                    (train, test, "%d-%d" % (repeat_nr, fold_nr))
                )

    def __iter__(self):
        """Yields group ids of training items, testing items, and the iteration id."""
        for train_ids, test_ids, iteration_id in self.iterations:
            yield train_ids, test_ids, iteration_id


def build_kfold_reference_dataset(corpus):
    """
    the more sophisticated CV splitter needs a good labeling to produce
    similar folds. I want to use the role+complextype labelling here,
    which is not in the datafiles loaded above. I thus extract it from
    the corpus of files directly.
    """
    data = {
        tid: graph.get_role_type_labels().values()
        for tid, graph in corpus.items()
    }
    return data

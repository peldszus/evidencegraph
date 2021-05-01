#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: Andreas Peldszus
"""


import json
import os
from itertools import chain, combinations
from collections import defaultdict
from numpy import mean
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

from .argtree import ArgTree
from .argtree import FULL_RELATION_SET
from .corpus import GraphCorpus, CORPORA
from .result_collector import ResultCollector


def evaluate(ground_truth, prediction):
    """
    Evaluate predictions against ground truth. Calculates several metrics
    including accuracy, macro-, micro- and weighted averaged precision,
    recall and f1, a confusion matrix, Cohen's and Fleiss' kappa, as well
    as class-wise scores.

    >>> d = evaluate([0,0,1,1],[0,0,1,1])
    >>> sorted(d.items())
    [('accuracy', 1.0),
     ('classwise', {'0': {'precision': 1.0, 'recall': 1.0, 'fscore': 1.0},
                    '1': {'precision': 1.0, 'recall': 1.0, 'fscore': 1.0}}),
     ('confusions', {0: {0: 2, 1: 0}, 1: {0: 0, 1: 2}}),
     ('k_cohen', {'k': 1.0, 'AE': 0.5, 'AO': 1.0}),
     ('k_fleiss', {'k': 1.0, 'AE': 0.5, 'AO': 1.0}),
     ('macro_avg', {'precision': 1.0, 'recall': 1.0, 'fscore': 1.0}),
     ('micro_avg', {'precision': 1.0, 'recall': 1.0, 'fscore': 1.0}),
     ('true_cat_dist', [2, 2]),
     ('weigh_avg', {'precision': 1.0, 'recall': 1.0, 'fscore': 1.0})]

    >>> d = evaluate([0,1,1,2,2,2],[0,0,2,2,2,2])
    >>> sorted(d.items())
    [('accuracy', 0.666...),
     ('classwise', {'0': {'precision': 0.5, 'recall': 1.0, 'fscore': 0.666...},
                    '1': {'precision': 0.0, 'recall': 0.0, 'fscore': 0.0},
                    '2': {'precision': 0.75, 'recall': 1.0, 'fscore': 0.857...}}),
     ('confusions', {0: {0: 1, 1: 0, 2: 0}, 1: {0: 1, 1: 0, 2: 1}, 2: {0: 0, 1: 0, 2: 3}}),
     ('k_cohen', {'k': 0.454..., 'AE': 0.388..., 'AO': 0.666...}),
     ('k_fleiss', {'k': 0.414..., 'AE': 0.430..., 'AO': 0.666...}),
     ('macro_avg', {'precision': 0.416..., 'recall': 0.666..., 'fscore': 0.507...}),
     ('micro_avg', {'precision': 0.666..., 'recall': 0.666..., 'fscore': 0.666...}),
     ('true_cat_dist', [1, 2, 3]),
     ('weigh_avg', {'precision': 0.458..., 'recall': 0.666..., 'fscore': 0.539...})]

    TODO: This still throws an error when calculating classwise predictions
    >> d = evaluate([1,2],[1,2])
    """

    def prfs_to_dict(l):
        return {"precision": l[0], "recall": l[1], "fscore": l[2]}

    results = {}
    items_count = len(ground_truth)

    # accuracy
    accuracy = accuracy_score(ground_truth, prediction)
    results["accuracy"] = accuracy

    # confusion matrix
    categories = set(ground_truth) | set(prediction)
    confusions = {
        gold: {pred: 0 for pred in categories} for gold in categories
    }
    for g, p in zip(ground_truth, prediction):
        confusions[g][p] += 1
    results["confusions"] = confusions

    # class wise precision, recall & f1
    classwise = precision_recall_fscore_support(
        ground_truth, prediction, average=None, warn_for=()
    )
    results["true_cat_dist"] = list(classwise[-1])
    results["classwise"] = {
        str(cl): prfs_to_dict(
            [classwise[0][cl], classwise[1][cl], classwise[2][cl]]
        )
        for cl in categories
    }

    # average precision, recall & f1
    results["macro_avg"] = prfs_to_dict(
        precision_recall_fscore_support(
            ground_truth,
            prediction,
            average="macro",
            pos_label=None,
            warn_for=(),
        )
    )
    results["micro_avg"] = prfs_to_dict(
        precision_recall_fscore_support(
            ground_truth,
            prediction,
            average="micro",
            pos_label=None,
            warn_for=(),
        )
    )
    results["weigh_avg"] = prfs_to_dict(
        precision_recall_fscore_support(
            ground_truth,
            prediction,
            average="weighted",
            pos_label=None,
            warn_for=(),
        )
    )

    # marginals
    gold_category_distribution = {
        g: sum([confusions[g][p] for p in categories]) for g in categories
    }
    pred_category_distribution = {
        p: sum([confusions[g][p] for g in categories]) for p in categories
    }

    # kappa
    expected_agreement_fleiss = sum(
        [
            (
                (gold_category_distribution[c] + pred_category_distribution[c])
                / (2.0 * items_count)
            )
            ** 2
            for c in categories
        ]
    )
    expected_agreement_cohen = sum(
        [
            (float(gold_category_distribution[c]) / items_count)
            * (float(pred_category_distribution[c]) / items_count)
            for c in categories
        ]
    )
    kappa_fleiss = (
        1.0
        * (accuracy - expected_agreement_fleiss)
        / (1 - expected_agreement_fleiss)
    )
    kappa_cohen = (
        1.0
        * (accuracy - expected_agreement_cohen)
        / (1 - expected_agreement_cohen)
    )
    results["k_fleiss"] = {
        "k": kappa_fleiss,
        "AE": expected_agreement_fleiss,
        "AO": accuracy,
    }
    results["k_cohen"] = {
        "k": kappa_cohen,
        "AE": expected_agreement_cohen,
        "AO": accuracy,
    }

    return results


def labelled_attachment(gold_trees, pred_trees):
    """
    Calculates labelled attachment score (LAS).

    Args:
        gold_trees (list): a list of ArgTree representing the ground truth
        pred_trees (list): a list of ArgTree representing the predictions

    Returns:
        labelled attachment score

    >>> g = ArgTree(from_triples=[(1, 2, 'sup'), (3, 2, 'att'), (4, 3, 'att')])
    >>> p = ArgTree(from_triples=[(1, 2, 'sup'), (3, 2, 'sup'), (4, 2, 'sup')])
    >>> labelled_attachment([g], [p])
    0.5
    """
    count_match, count_total = 0, 0
    for gold, pred in zip(gold_trees, pred_trees):
        triples_pairs = zip(
            gold.get_triples(include_root=True),
            pred.get_triples(include_root=True),
        )
        for (g_src, g_trg, g_rel), (p_src, p_trg, p_rel) in triples_pairs:
            assert g_src == p_src
            count_total += 1
            if g_trg == p_trg and g_rel == p_rel:
                count_match += 1
    if count_match == 0 or count_total == 0:
        return 0.0
    else:
        return float(count_match) / count_total


def eval_prediction(gold_trees, pred_trees):
    """
    Evaluates predicted structures against ground truth structures.

    Args:
        gold_trees (list): a list of ArgTree representing the ground truth
        pred_trees (list): a list of ArgTree representing the predictions

    Returns:
        a list of the level-specific evaluation scores for the levels

    >>> g = ArgTree(from_triples=[(1, 2, 'sup'), (3, 2, 'att'), (4, 3, 'att')])
    >>> p = ArgTree(from_triples=[(1, 2, 'sup'), (3, 2, 'att'), (4, 2, 'sup')])
    >>> l = eval_prediction([g], [p])
    >>> [(level, scores['accuracy']) for level, scores in l]
    [('cc', 1.0), ('ro', 1.0), ('fu', 0.75), ('at', 0.833...), ('lat', 0.75)]
    >>> g = ArgTree(from_triples=[(2, 1, 'sup'), (3, 2, 'sup')])
    >>> p = ArgTree(from_triples=[(2, 1, 'sup'), (3, 1, 'sup')])
    >>> l = eval_prediction([g], [p])
    >>> [(level, scores['accuracy']) for level, scores in l]
    [('cc', 1.0), ('ro', 1.0), ('fu', 1.0), ('at', 0.666...), ('lat', 0.666...)]
    """
    assert [t.get_tid() for t in gold_trees] == [
        t.get_tid() for t in pred_trees
    ]
    gold_vectors = [t.get_vector() for t in gold_trees]
    pred_vectors = [t.get_vector() for t in pred_trees]

    def score_it(gold, pred, level):
        level_gold = list(chain.from_iterable([v[level] for v in gold]))
        level_pred = list(chain.from_iterable([v[level] for v in pred]))
        assert len(level_gold) == len(level_pred), "{} {} {}".format(
            level, level_gold, level_pred
        )
        return evaluate(level_gold, level_pred)

    results = []
    for level in ["cc", "ro", "fu", "at"]:
        results.append((level, score_it(gold_vectors, pred_vectors, level)))

    lat = labelled_attachment(gold_trees, pred_trees)
    results.append(("lat", {"accuracy": lat}))
    return results


def filter_triples(triples):
    """
    Maps relation names in triples to their short names.

    >>> list(filter_triples([(1, 2, 'support'), (3, 2, 'attack')]))
    [(1, 2, 'sup'), (3, 2, 'att')]
    """
    mapped = {"support": "sup", "attack": "att"}
    for src, trg, rel in triples:
        yield src, trg, mapped[rel]


def load_predictions(path, replace_rel=None, relation_set=FULL_RELATION_SET):
    """
    Loads the predictions from a json file.

    The json file is expected to have the following structure:
    {iteration_id: {text_id: [edges]}}

    Args:
        path (str): path of the json file to load
        replace_relations (optional, dict): a dict mapping input relation
            names to output relation names

    Returns:
        a dictionary with the structure {iteration_id: {text_id: ArgTree}}
    """
    raw = json.load(open(path))
    predictions = defaultdict(dict)
    for iteration, texts in raw.items():
        for tid, triples in texts.items():
            if replace_rel:
                triples = [
                    (s, t, replace_rel.get(r, r)) for s, t, r in triples
                ]
            t = ArgTree(
                from_triples=triples, text_id=tid, relation_set=relation_set
            )
            predictions[iteration][tid] = t
    return predictions


def evaluate_iterations(predictions, gold, result_collector, condition):
    """
    Scores the predicted structures of each iteration against the gold
    structures as saves the results (as a side effect) into the
    result_collector under the specified condition identifier.

    Args:
        predictions (dict): the predicted structures of the form
            {iteration_id: {text_id: ArgTree}}
        gold (dict): the ground truth structures of the form
            {text_id: ArgTree}
        result_collector (ResultCollector): the collector to store results in
        condition (string): the condition identifier to store the result for

    """
    for iteration_id, texts in predictions.items():
        texts_in_iteration = sorted(texts.keys())
        gold_trees = [gold[tid] for tid in texts_in_iteration]
        pred_trees = [texts[tid] for tid in texts_in_iteration]
        for level, scores in eval_prediction(gold_trees, pred_trees):
            result_collector.add_result(condition, iteration_id, level, scores)


def print_scores(result_collector):
    """
    Print the collected results for all conditions and all levels in three
    metrics: Cohen's k, macro avg. F1, and positive attachment F1.

    Args:
        result_collector (ResultCollector): the collected results
    """
    # print("\n# Metric: Cohen's kappa")
    # result_collector.set_metric(['k_cohen', 'k'])
    # result_collector.print_all_results()
    print("\n# Metric: Macro avg. F1")
    result_collector.set_metric(["macro_avg", "fscore"])
    # result_collector.print_all_results()
    result_collector.print_result_for_level("cc")
    result_collector.print_result_for_level("ro", print_header=False)
    result_collector.print_result_for_level("fu", print_header=False)
    result_collector.print_result_for_level("at", print_header=False)

    # print("\nMetric: Positive attachment F1")
    # result_collector.set_metric(['classwise', '1', 'fscore'])
    # result_collector.print_result_for_level('at')
    print("\n# Metric: Labelled attachment score")
    result_collector.set_metric(["accuracy"])
    result_collector.print_result_for_level("lat")


def print_significance(result_collector, conditionA, conditionB, levels=[]):
    """
    Test and print the significance of difference between the results for
    `conditionA` and `conditionB` on the specified levels. Macro avg. F1 is
    used as the scoring metric. The pvalue of the tests is printed.

    Args:
        result_collector (ResultCollector): the collected results
        conditionA (string): condition to compare
        conditionB (string): condition to compare
        levels (list): the levels to do the significance testing on
    """
    print(
        "\n# Testing significance of difference: {} vs. {}".format(
            conditionA, conditionB
        )
    )
    result_collector.set_metric(["macro_avg", "fscore"])
    print("level\tp_value")
    for level in levels:
        _, pvalue = result_collector.wilcoxon(conditionA, conditionB, level)
        print("{}\t{:.5f}".format(level, pvalue))

    level = "lat"
    result_collector.set_metric(["accuracy"])
    _, pvalue = result_collector.wilcoxon(conditionA, conditionB, level)
    print("{}\t{:.5f}".format(level, pvalue))


def error_analysis(predictions, gold, result_collector):
    """
    Aggregates evaluation scores of single text predictions over iterations
    """
    # scores = defaultdict(list)
    for iteration_id, texts in predictions.items():
        # map iteration id to fold
        fold = str(int(iteration_id) / 5)
        for tid, pred_tree in texts.items():
            gold_tree = gold[tid]
            print(iteration_id, fold, tid)
            print(gold_tree.get_triples())
            print(pred_tree.get_triples())
            for level, scores in eval_prediction([gold_tree], [pred_tree]):
                result_collector.add_result(tid, fold, level, scores)
    print("Done.")


def class_scores(result_collector, level):
    print("\n# Classwise scores (P, R, F1) for level {}".format(level))
    result_collector.set_metric(["classwise"], ignore_type=True)
    d = defaultdict(dict)
    for condition in result_collector.conditions:
        results = result_collector._get_result(condition, level)
        if None in results:
            print("Warning: Classwise results not available. Fix bugs!")
            return
        classes = sorted(
            set(key for result in results for key in result.keys())
        )
        for class_ in classes:
            p = mean([result[class_]["precision"] for result in results])
            r = mean([result[class_]["recall"] for result in results])
            f = mean([result[class_]["fscore"] for result in results])
            d[class_][condition] = (p, r, f)

    print("\t".join(["condition"] + result_collector.conditions))
    for class_ in sorted(d.keys()):
        line = "{}".format(class_)
        for condition in result_collector.conditions:
            p, r, f = d[class_][condition]
            line += "\t{:.3f} {:.3f} {:.3f}".format(p, r, f)
        print(line)


def evaluate_setting(
    language,
    segmentation,
    relationset,
    conditions,
    corpus_id=None,
    predictions_path=None,
):
    levels = ["cc", "ro", "fu", "at"]
    rc = ResultCollector()

    print(
        "\n\nEVALUATING SETTING {}, {}, {}:".format(
            language, segmentation, relationset.functions
        )
    )

    # load gold corpus
    gc = GraphCorpus()
    gc.load(CORPORA[corpus_id]["path"])
    gold = gc.trees(segmentation, relationset)

    if not predictions_path:
        predictions_path = "data"

    for run in conditions:
        p = load_predictions(
            os.path.join(predictions_path, "{}.json".format(run)),
            relation_set=relationset,
        )
        evaluate_iterations(p, gold, rc, run)

    print_scores(rc)

    for level in levels:
        class_scores(rc, level)

    for condition_1, condition_2 in combinations(conditions, 2):
        print_significance(rc, condition_1, condition_2, levels=levels)

    return rc

#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

"""
@author: Andreas Peldszus
"""

import json

from evidencegraph.argtree import RELATION_SETS_BY_NAME
from evidencegraph.corpus import GraphCorpus, CORPORA
from evidencegraph.evaluation import evaluate_setting
from evidencegraph.experiment import run_experiment_condition
from evidencegraph.features_text import init_language
from evidencegraph.folds import get_static_folds


def test_experiment_workflow():
    # set parameters of the tiny experiment
    corpus_name = "m112en"
    corpus = GraphCorpus()
    corpus.load(CORPORA[corpus_name]["path"])
    language = CORPORA[corpus_name]["language"]
    feature_set = [
        "default",
        "bow",
        "bow_2gram",
        "first_three",
        "tags",
        "deps_lemma",
        "deps_tag",
        "punct",
        "verb_main",
        "verb_all",
        "discourse_marker",
        "context",
        "clusters",
        "clusters_2gram",
        "discourse_relation",
        "vector_left_right",
        "vector_source_target",
        "verb_segment",
        "same_sentence",
        "matrix_clause",
    ]
    segmentation = "adu"
    condition_name = "workflow_test"
    params = {
        "relation_set": "SIMPLE_RELATION_SET",
        "optimize": False,
        "optimize_weighting": False,
    }

    # run the tiny experiment
    predictions_path = "/tmp"
    features = init_language(language)
    folds = list(get_static_folds())[:2]
    features.feature_set = feature_set
    params["relation_set"] = RELATION_SETS_BY_NAME.get(params["relation_set"])
    texts, trees = corpus.segments_trees(segmentation, params["relation_set"])
    predictions, _decisions = run_experiment_condition(
        texts, trees, folds, features, params, condition_name
    )
    with open("{}/{}.json".format(predictions_path, condition_name), "w") as f:
        json.dump(predictions, f, indent=1, sort_keys=True)

    # evaluate the tiny experiment
    result_collector = evaluate_setting(
        language,
        segmentation,
        params["relation_set"],
        [condition_name],
        corpus_id=corpus_name,
        predictions_path=predictions_path,
    )
    assert result_collector
    result_collector.set_metric(["macro_avg", "fscore"])
    assert result_collector._sum_result(condition_name, "cc")["mean"] >= 0.8
    assert result_collector._sum_result(condition_name, "ro")["mean"] >= 0.7
    assert result_collector._sum_result(condition_name, "fu")["mean"] >= 0.7
    assert result_collector._sum_result(condition_name, "at")["mean"] >= 0.6

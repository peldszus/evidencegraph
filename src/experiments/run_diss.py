#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

"""
@author: Andreas Peldszus
"""


import json

from evidencegraph.argtree import RELATION_SETS_BY_NAME
from evidencegraph.corpus import GraphCorpus, CORPORA
from evidencegraph.experiment import run_experiment_condition
from evidencegraph.features_text import init_language
from evidencegraph.folds import get_static_folds


if __name__ == "__main__":
    # define features sets
    features_old = [
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
    ]
    features_new_non_vector = [
        "clusters",
        "clusters_2gram",
        "discourse_relation",
        "vector_left_right",
        "vector_source_target",
        "verb_segment",
        "same_sentence",
        "matrix_clause",
    ]
    features_all_but_vector = features_old + features_new_non_vector

    conditions = {
        # de adu
        "m112de-diss-adu-simple-op|equal": {
            "corpus": "m112de",
            "feature_set": features_all_but_vector,
            "relation_set": "SIMPLE_RELATION_SET",
            "segmentation": "adu",
            "optimize": True,
            "optimize_weighting": False,
        },
        "m112de-diss-adu-full-op|equal": {
            "corpus": "m112de",
            "feature_set": features_all_but_vector,
            "relation_set": "FULL_RELATION_SET_ADU",
            "segmentation": "adu",
            "optimize": True,
            "optimize_weighting": False,
        },
        "m112de-diss-adu-simple-op|train": {
            "corpus": "m112de",
            "feature_set": features_all_but_vector,
            "relation_set": "SIMPLE_RELATION_SET",
            "segmentation": "adu",
            "optimize": True,
            "optimize_weighting": "train",
        },
        "m112de-diss-adu-full-op|train": {
            "corpus": "m112de",
            "feature_set": features_all_but_vector,
            "relation_set": "FULL_RELATION_SET_ADU",
            "segmentation": "adu",
            "optimize": True,
            "optimize_weighting": "train",
        },
        "m112de-diss-adu-simple-op|test": {
            "corpus": "m112de",
            "feature_set": features_all_but_vector,
            "relation_set": "SIMPLE_RELATION_SET",
            "segmentation": "adu",
            "optimize": True,
            "optimize_weighting": "inner_cv",
        },
        "m112de-diss-adu-full-op|test": {
            "corpus": "m112de",
            "feature_set": features_all_but_vector,
            "relation_set": "FULL_RELATION_SET_ADU",
            "segmentation": "adu",
            "optimize": True,
            "optimize_weighting": "inner_cv",
        },
        # en adu
        "m112en-diss-adu-simple-op|equal": {
            "corpus": "m112en",
            "feature_set": features_all_but_vector,
            "relation_set": "SIMPLE_RELATION_SET",
            "segmentation": "adu",
            "optimize": True,
            "optimize_weighting": False,
        },
        "m112en-diss-adu-full-op|equal": {
            "corpus": "m112en",
            "feature_set": features_all_but_vector,
            "relation_set": "FULL_RELATION_SET_ADU",
            "segmentation": "adu",
            "optimize": True,
            "optimize_weighting": False,
        },
        "m112en-diss-adu-simple-op|train": {
            "corpus": "m112en",
            "feature_set": features_all_but_vector,
            "relation_set": "SIMPLE_RELATION_SET",
            "segmentation": "adu",
            "optimize": True,
            "optimize_weighting": "train",
        },
        "m112en-diss-adu-full-op|train": {
            "corpus": "m112en",
            "feature_set": features_all_but_vector,
            "relation_set": "FULL_RELATION_SET_ADU",
            "segmentation": "adu",
            "optimize": True,
            "optimize_weighting": "train",
        },
        "m112en-diss-adu-simple-op|test": {
            "corpus": "m112en",
            "feature_set": features_all_but_vector,
            "relation_set": "SIMPLE_RELATION_SET",
            "segmentation": "adu",
            "optimize": True,
            "optimize_weighting": "inner_cv",
        },
        "m112en-diss-adu-full-op|test": {
            "corpus": "m112en",
            "feature_set": features_all_but_vector,
            "relation_set": "FULL_RELATION_SET_ADU",
            "segmentation": "adu",
            "optimize": True,
            "optimize_weighting": "inner_cv",
        },
        # en edu
        "m112en-diss-edu-simple-op|equal": {
            "corpus": "m112en_fine",
            "feature_set": features_all_but_vector,
            "relation_set": "SIMPLE_RELATION_SET",
            "segmentation": "edu",
            "optimize": True,
            "optimize_weighting": False,
        },
        "m112en-diss-edu-full-op|equal": {
            "corpus": "m112en_fine",
            "feature_set": features_all_but_vector,
            "relation_set": "FULL_RELATION_SET",
            "segmentation": "edu",
            "optimize": True,
            "optimize_weighting": False,
        },
        "m112en-diss-edu-simple-op|train": {
            "corpus": "m112en_fine",
            "feature_set": features_all_but_vector,
            "relation_set": "SIMPLE_RELATION_SET",
            "segmentation": "edu",
            "optimize": True,
            "optimize_weighting": "train",
        },
        "m112en-diss-edu-full-op|train": {
            "corpus": "m112en_fine",
            "feature_set": features_all_but_vector,
            "relation_set": "FULL_RELATION_SET",
            "segmentation": "edu",
            "optimize": True,
            "optimize_weighting": "train",
        },
        "m112en-diss-edu-simple-op|test": {
            "corpus": "m112en_fine",
            "feature_set": features_all_but_vector,
            "relation_set": "SIMPLE_RELATION_SET",
            "segmentation": "edu",
            "optimize": True,
            "optimize_weighting": "inner_cv",
        },
        "m112en-diss-edu-full-op|test": {
            "corpus": "m112en_fine",
            "feature_set": features_all_but_vector,
            "relation_set": "FULL_RELATION_SET",
            "segmentation": "edu",
            "optimize": True,
            "optimize_weighting": "inner_cv",
        },
    }

    # run all experiment conditions
    folds = list(get_static_folds())
    for condition_name, params in conditions.items():
        print("### Running experiment condition", condition_name)

        # load corpus
        corpus_name = params.pop("corpus")
        corpus = GraphCorpus()
        corpus.load(CORPORA[corpus_name]["path"])

        # set condition params
        language = CORPORA[corpus_name]["language"]
        features = init_language(language)
        features.feature_set = params.pop("feature_set")
        params["relation_set"] = RELATION_SETS_BY_NAME.get(
            params["relation_set"]
        )
        texts, trees = corpus.segments_trees(
            params.pop("segmentation"), params["relation_set"]
        )

        # run condition
        predictions, _decisions = run_experiment_condition(
            texts, trees, folds, features, params, condition_name
        )
        with open("data/{}.json".format(condition_name), "w") as f:
            json.dump(predictions, f, indent=1, sort_keys=True)

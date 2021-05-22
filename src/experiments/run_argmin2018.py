"""
@author: Andreas Peldszus
"""


import json

from evidencegraph.argtree import RELATION_SETS_BY_NAME
from evidencegraph.corpus import combine_corpora
from evidencegraph.experiment import run_experiment_condition
from evidencegraph.features_text import init_language


if __name__ == "__main__":
    language = "en"

    # features sets
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
        # evaluate on old corpus (m112en)
        "argmin2018-normal-m112en-adu-simple-op|equal": {
            "corpora": ["m112en"],
            "mode": "normal",
            "feature_set": features_all_but_vector,
            "relation_set": "SIMPLE_RELATION_SET",
            "segmentation": "adu",
            "optimize": True,
            "optimize_weighting": False,
        },
        "argmin2018-cross-part2+m112en-adu-simple-op|equal": {
            "corpora": ["m112en_part2", "m112en"],
            "mode": "cross",
            "feature_set": features_all_but_vector,
            "relation_set": "SIMPLE_RELATION_SET",
            "segmentation": "adu",
            "optimize": True,
            "optimize_weighting": False,
        },
        "argmin2018-add-part2+m112en-adu-simple-op|equal": {
            "corpora": ["m112en_part2", "m112en"],
            "mode": "add",
            "feature_set": features_all_but_vector,
            "relation_set": "SIMPLE_RELATION_SET",
            "segmentation": "adu",
            "optimize": True,
            "optimize_weighting": False,
        },
        # evaluate on new corpus (m112en_part2)
        "argmin2018-normal-part2-adu-simple-op|equal": {
            "corpora": ["m112en_part2"],
            "mode": "normal",
            "feature_set": features_all_but_vector,
            "relation_set": "SIMPLE_RELATION_SET",
            "segmentation": "adu",
            "optimize": True,
            "optimize_weighting": False,
        },
        "argmin2018-cross-m112en+part2-adu-simple-op|equal": {
            "corpora": ["m112en", "m112en_part2"],
            "mode": "cross",
            "feature_set": features_all_but_vector,
            "relation_set": "SIMPLE_RELATION_SET",
            "segmentation": "adu",
            "optimize": True,
            "optimize_weighting": False,
        },
        "argmin2018-add-m112en+part2-adu-simple-op|equal": {
            "corpora": ["m112en", "m112en_part2"],
            "mode": "add",
            "feature_set": features_all_but_vector,
            "relation_set": "SIMPLE_RELATION_SET",
            "segmentation": "adu",
            "optimize": True,
            "optimize_weighting": False,
        },
    }

    # run all experiment conditions
    for condition_name, params in conditions.items():
        print("### Running experiment condition", condition_name)

        # load and combine corpora
        corpus, folds = combine_corpora(
            params.pop("corpora"), mode=params.pop("mode")
        )

        # set condition params
        features = init_language(language)
        features.feature_set = params.pop("feature_set")
        params["relation_set"] = RELATION_SETS_BY_NAME.get(
            params["relation_set"]
        )
        texts, trees = corpus.segments_trees(
            params.pop("segmentation"), params["relation_set"]
        )

        # health check
        for tid in sorted(trees):
            triples = trees[tid].get_triples(include_root=True)
            assert len(triples) == len(
                texts[tid]
            ), "Somethings wrong with {}".format(tid)

        # run condition
        predictions, _decisions = run_experiment_condition(
            texts, trees, folds, features, params, condition_name
        )
        with open("data/{}.json".format(condition_name), "w") as f:
            json.dump(predictions, f, indent=1, sort_keys=True)

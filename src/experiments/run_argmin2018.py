#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

'''
@author: Andreas Peldszus
'''


import json

from evidencegraph.argtree import RELATION_SETS_BY_NAME
from evidencegraph.corpus import GraphCorpus, CORPORA
from evidencegraph.experiment import run_experiment_condition
from evidencegraph.features_text import init_language


def create_corpus(corpora, mode="normal"):
    """
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
            gc.load(CORPORA[corpus]['path'])
        folds = list(gc.create_folds())

    elif mode == "cross":
        gc = GraphCorpus()
        assert len(corpora) > 1
        last = corpora[-1]
        gc.load(CORPORA[last]['path'])
        last_folds = list(gc.create_folds())
        first_corpora_text_ids = []
        for corpus in corpora[:-1]:
            ids = gc.load(CORPORA[corpus]['path'])
            first_corpora_text_ids.extend(ids)
        folds = [(first_corpora_text_ids, test, n)
                 for _, test, n in last_folds]

    elif mode == "add":
        gc = GraphCorpus()
        assert len(corpora) > 1
        last = corpora[-1]
        all_corpora_text_ids = []
        ids = gc.load(CORPORA[last]['path'])
        all_corpora_text_ids.extend(ids)
        last_folds = list(gc.create_folds())
        for corpus in corpora[:-1]:
            ids = gc.load(CORPORA[corpus]['path'])
            all_corpora_text_ids.extend(ids)
        folds = [([i for i in all_corpora_text_ids if i not in test], test, n)
                 for _, test, n in last_folds]

    return gc, folds


if __name__ == '__main__':
    language = "en"

    # features sets
    features_old = [
        'default', 'bow', 'bow_2gram', 'first_three',
        'tags', 'deps_lemma', 'deps_tag',
        'punct', 'verb_main', 'verb_all', 'discourse_marker',
        'context'
    ]
    features_new_non_vector = [
        'clusters', 'clusters_2gram', 'discourse_relation',
        'vector_left_right', 'vector_source_target',
        'verb_segment', 'same_sentence', 'matrix_clause'
    ]
    features_all_but_vector = features_old + features_new_non_vector

    conditions = {
        # evaluate on old corpus (m112en)
        'argmin2018-normal-m112en-adu-simple-op|equal': {
            'corpora': ['m112en'],
            'mode': 'normal',
            'feature_set': features_all_but_vector,
            'relation_set': "SIMPLE_RELATION_SET",
            'segmentation': 'adu',
            'optimize': True,
            'optimize_weighting': False
        },
        'argmin2018-cross-part2+m112en-adu-simple-op|equal': {
            'corpora': ['m112en_part2', 'm112en'],
            'mode': 'cross',
            'feature_set': features_all_but_vector,
            'relation_set': "SIMPLE_RELATION_SET",
            'segmentation': 'adu',
            'optimize': True,
            'optimize_weighting': False
        },
        'argmin2018-add-part2+m112en-adu-simple-op|equal': {
            'corpora': ['m112en_part2', 'm112en'],
            'mode': 'add',
            'feature_set': features_all_but_vector,
            'relation_set': "SIMPLE_RELATION_SET",
            'segmentation': 'adu',
            'optimize': True,
            'optimize_weighting': False
        },

        # evaluate on new corpus (m112en_part2)
        'argmin2018-normal-part2-adu-simple-op|equal': {
            'corpora': ['m112en_part2'],
            'mode': 'normal',
            'feature_set': features_all_but_vector,
            'relation_set': "SIMPLE_RELATION_SET",
            'segmentation': 'adu',
            'optimize': True,
            'optimize_weighting': False
        },
        'argmin2018-cross-m112en+part2-adu-simple-op|equal': {
            'corpora': ['m112en', 'm112en_part2'],
            'mode': 'cross',
            'feature_set': features_all_but_vector,
            'relation_set': "SIMPLE_RELATION_SET",
            'segmentation': 'adu',
            'optimize': True,
            'optimize_weighting': False
        },
        'argmin2018-add-m112en+part2-adu-simple-op|equal': {
            'corpora': ['m112en', 'm112en_part2'],
            'mode': 'add',
            'feature_set': features_all_but_vector,
            'relation_set': "SIMPLE_RELATION_SET",
            'segmentation': 'adu',
            'optimize': True,
            'optimize_weighting': False
        }
    }

    # run all experiment conditions
    for condition_name, params in conditions.items():
        print "### Running experiment condition", condition_name

        # load and combine corpora
        corpus, folds = create_corpus(
            params.pop('corpora'), mode=params.pop('mode'))

        # set condition params
        features = init_language(language)
        features.feature_set = params.pop('feature_set')
        params['relation_set'] = RELATION_SETS_BY_NAME.get(
            params['relation_set'])
        texts, trees = corpus.segments_trees(
            params.pop('segmentation'), params['relation_set'])

        # health check
        for tid in sorted(trees):
            triples = trees[tid].get_triples(include_root=True)
            assert len(triples) == len(texts[tid]), "Somethings wrong with {}".format(tid)

        # run condition
        predictions, _decisions = run_experiment_condition(
            texts, trees, folds, features, params, condition_name)
        with open('data/{}.json'.format(condition_name), 'w') as f:
            json.dump(predictions, f, indent=1, sort_keys=True)

#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

'''
@author: Andreas Peldszus
'''

import json
from argparse import ArgumentParser

from evidencegraph.argtree import RELATION_SETS_BY_NAME
from evidencegraph.corpus import GraphCorpus, CORPORA
from evidencegraph.experiment import run_experiment_condition
from evidencegraph.features_text import init_language
from evidencegraph.folds import get_static_folds


if __name__ == '__main__':
    parser = ArgumentParser(description=("Learn"))
    parser.add_argument('--corpus', '-c', choices=CORPORA.keys(), default='m112en',
                        help='the corpus to train on')
    args = parser.parse_args()
    corpus_name = args.corpus
    corpus = GraphCorpus()
    corpus.load(CORPORA[corpus_name]['path'])
    language = CORPORA[corpus_name]['language']

    feature_set = [
        'default', 'bow', 'bow_2gram', 'first_three',
        'tags', 'deps_lemma', 'deps_tag',
        'punct', 'verb_main', 'verb_all', 'discourse_marker',
        'context', 'clusters', 'clusters_2gram', 'discourse_relation',
        'vector_left_right', 'vector_source_target',
        'verb_segment', 'same_sentence', 'matrix_clause'
    ]

    # define experiment conditions
    conditions = {
        '{}-test-adu-simple-noop|equal'.format(corpus_name): {
            'feature_set': feature_set,
            'relation_set': "SIMPLE_RELATION_SET",
            'segmentation': 'adu',
            'optimize': False,
            'optimize_weighting': False
        }
    }

    # run all experiment conditions
    features = init_language(language)
    folds = list(get_static_folds())
    folds = folds[:5]  # remove this to run all 50 train/test splits
    for condition_name, params in conditions.items():
        features.feature_set = params.pop('feature_set')
        params['relation_set'] = RELATION_SETS_BY_NAME.get(params['relation_set'])
        texts, trees = corpus.segments_trees(params.pop('segmentation'), params['relation_set'])
        print "### Running experiment condition", condition_name
        predictions, _decisions = run_experiment_condition(texts, trees, folds, features, params, condition_name)
        with open('data/{}.json'.format(condition_name), 'w') as f:
            json.dump(predictions, f, indent=1, sort_keys=True)

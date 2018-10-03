#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

'''
@author: Andreas Peldszus
'''

import json
import os
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
from numpy import mean

from evidencegraph.argtree import SIMPLE_RELATION_SET
from evidencegraph.classifiers import EvidenceGraphClassifier
from evidencegraph.features_text import (feature_function_segmentpairs,
                                         feature_function_segments,
                                         init_language)
from evidencegraph.folds import get_static_folds
from evidencegraph.utils import load_corpus, hash_of_featureset, load, save


modelpath = "data/models/"


def folds_static(in_corpus, out_corpus, params, condition_name):
    maF1s = defaultdict(list)
    miF1s = defaultdict(list)
    folds = list(get_static_folds())
    predictions = defaultdict(dict)
    decisions = defaultdict(dict)

    for train_tids, test_tids, i in folds[:5]:
        print "Iteration: {}".format(i)
        ensemble_basename = condition_name.split('|')[0]
        ensemble_name = "{}__{}__{}".format(
            ensemble_basename, hash_of_featureset(params['feature_set']), i)
        clf = EvidenceGraphClassifier(
                feature_function_segments, feature_function_segmentpairs, **params)
        train_txt = [g for t, g in in_corpus.iteritems() if t in train_tids]
        train_arg = [g for t, g in out_corpus.iteritems() if t in train_tids]
        try:
            # load ensemble of pretrained base classifiers
            clf.ensemble = load(modelpath + ensemble_name)
            if params['optimize_weighting']:
                # and train metaclassifier (if desired)
                clf.train_metaclassifier(train_txt, train_arg)
        except RuntimeError:
            # train ensemble
            clf.train(train_txt, train_arg)
            save(clf.ensemble, modelpath + ensemble_name)

        # test
        test_txt = [g for t, g in in_corpus.iteritems() if t in test_tids]
        test_arg = [g for t, g in out_corpus.iteritems() if t in test_tids]
        score_msg = ''
        for level, base_classifier in clf.ensemble.items():
            maF1, miF1 = base_classifier.test(test_txt, test_arg)
            maF1s[level].append(maF1)
            miF1s[level].append(miF1)
            score_msg += "{}: {:.3f}\t".format(level, maF1)
        decoded_scores = []
        for t in test_tids:
            mst = clf.predict(in_corpus[t])
            decoded_scores.append(clf.score(mst, out_corpus[t]))
            predictions[i][t] = mst.get_triples()
            decisions[i][t] = clf.predict_decisions(in_corpus[t])
        score_msg += "decoded: {:.3f}\t".format(mean(decoded_scores))
        print score_msg

    print "Average macro and micro F1:"
    for level in maF1s:
        avg_maF1 = mean(maF1s[level])
        avg_miF1 = mean(miF1s[level])
        print level, avg_maF1, avg_miF1

    return predictions, decisions


if __name__ == '__main__':
    parser = ArgumentParser(description=("Learn"))
    parser.add_argument('--lang', '-l', choices=['en', 'de'], default='en',
                        help='the language to consider the predictions of')
    args = parser.parse_args()
    language = args.lang
    corpus_name = 'm112{}'.format(language)

    features = [
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
            'feature_set': features,
            'relation_set': SIMPLE_RELATION_SET,
            'segmentation': 'adu',
            'optimize': False,
            'optimize_weighting': False
        }
    }

    # run all experiment conditions
    init_language(language)
    for condition_name, params in conditions.items():
        texts, trees = load_corpus(
            language, params.pop('segmentation'), params['relation_set'])
        print "### Running experiment condition", condition_name
        predictions, _decisions = folds_static(texts, trees, params, condition_name)
        with open('data/{}.json'.format(condition_name), 'w') as f:
            json.dump(predictions, f, indent=1, sort_keys=True)

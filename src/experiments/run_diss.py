#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

'''
@author: Andreas Peldszus
'''

import json
from collections import defaultdict
from datetime import datetime

from numpy import mean

from evidencegraph.argtree import RELATION_SETS_BY_NAME
from evidencegraph.classifiers import EvidenceGraphClassifier
from evidencegraph.corpus import GraphCorpus, CORPORA
from evidencegraph.features_text import init_language
from evidencegraph.folds import get_static_folds
from evidencegraph.utils import hash_of_featureset

modelpath = "data/models/"


def folds_static(in_corpus, out_corpus, folds, features, params, condition_name):
    maF1s = defaultdict(list)
    miF1s = defaultdict(list)
    predictions = defaultdict(dict)
    decisions = defaultdict(dict)

    for train_tids, test_tids, i in folds:
        print "[{}] Iteration: {}\t".format(datetime.now(), i)
        ensemble_basename = condition_name.split('|')[0]
        ensemble_name = "{}__{}__{}".format(
            ensemble_basename, hash_of_featureset(features.feature_set), i)
        clf = EvidenceGraphClassifier(
            features.feature_function_segments,
            features.feature_function_segmentpairs,
            **params
        )
        train_txt = [g for t, g in in_corpus.iteritems() if t in train_tids]
        train_arg = [g for t, g in out_corpus.iteritems() if t in train_tids]
        try:
            # load ensemble of pretrained base classifiers
            clf.load(modelpath + ensemble_name)
            if params['optimize_weighting']:
                # and train metaclassifier (if desired)
                clf.train_metaclassifier(train_txt, train_arg)
        except RuntimeError:
            # train ensemble
            clf.train(train_txt, train_arg)
            clf.save(modelpath + ensemble_name)

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
    # define features sets
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
        # de adu
        'm112de-diss-adu-simple-op|equal': {
            'corpus': 'm112de',
            'feature_set': features_all_but_vector,
            'relation_set': "SIMPLE_RELATION_SET",
            'segmentation': 'adu',
            'optimize': True,
            'optimize_weighting': False
        },
        'm112de-diss-adu-full-op|equal': {
            'corpus': 'm112de',
            'feature_set': features_all_but_vector,
            'relation_set': "FULL_RELATION_SET_ADU",
            'segmentation': 'adu',
            'optimize': True,
            'optimize_weighting': False
        },
        'm112de-diss-adu-simple-op|train': {
            'corpus': 'm112de',
            'feature_set': features_all_but_vector,
            'relation_set': "SIMPLE_RELATION_SET",
            'segmentation': 'adu',
            'optimize': True,
            'optimize_weighting': 'train'
        },
        'm112de-diss-adu-full-op|train': {
            'corpus': 'm112de',
            'feature_set': features_all_but_vector,
            'relation_set': "FULL_RELATION_SET_ADU",
            'segmentation': 'adu',
            'optimize': True,
            'optimize_weighting': 'train'
        },
        'm112de-diss-adu-simple-op|test': {
            'corpus': 'm112de',
            'feature_set': features_all_but_vector,
            'relation_set': "SIMPLE_RELATION_SET",
            'segmentation': 'adu',
            'optimize': True,
            'optimize_weighting': 'inner_cv'
        },
        'm112de-diss-adu-full-op|test': {
            'corpus': 'm112de',
            'feature_set': features_all_but_vector,
            'relation_set': "FULL_RELATION_SET_ADU",
            'segmentation': 'adu',
            'optimize': True,
            'optimize_weighting': 'inner_cv'
        },

        # en adu
        'm112en-diss-adu-simple-op|equal': {
            'corpus': 'm112en',
            'feature_set': features_all_but_vector,
            'relation_set': "SIMPLE_RELATION_SET",
            'segmentation': 'adu',
            'optimize': True, 
            'optimize_weighting': False
        }, # yes
        'm112en-diss-adu-full-op|equal': {
            'corpus': 'm112en',
            'feature_set': features_all_but_vector,
            'relation_set': "FULL_RELATION_SET_ADU",
            'segmentation': 'adu',
            'optimize': True,
            'optimize_weighting': False
        },
        'm112en-diss-adu-simple-op|train': {
            'corpus': 'm112en',
            'feature_set': features_all_but_vector,
            'relation_set': "SIMPLE_RELATION_SET",
            'segmentation': 'adu',
            'optimize': True,
            'optimize_weighting': 'train'
        },
        'm112en-diss-adu-full-op|train': {
            'corpus': 'm112en',
            'feature_set': features_all_but_vector,
            'relation_set': "FULL_RELATION_SET_ADU",
            'segmentation': 'adu',
            'optimize': True,
            'optimize_weighting': 'train'
        },
        'm112en-diss-adu-simple-op|test': {
            'corpus': 'm112en',
            'feature_set': features_all_but_vector,
            'relation_set': "SIMPLE_RELATION_SET",
            'segmentation': 'adu',
            'optimize': True,
            'optimize_weighting': 'inner_cv'
        },
        'm112en-diss-adu-full-op|test': {
            'corpus': 'm112en',
            'feature_set': features_all_but_vector,
            'relation_set': "FULL_RELATION_SET_ADU",
            'segmentation': 'adu',
            'optimize': True,
            'optimize_weighting': 'inner_cv'
        },

        # en edu
        'm112en-diss-edu-simple-op|equal': {
            'corpus': 'm112en_fine',
            'feature_set': features_all_but_vector,
            'relation_set': "SIMPLE_RELATION_SET",
            'segmentation': 'edu',
            'optimize': True,
            'optimize_weighting': False
        },
        'm112en-diss-edu-full-op|equal': {
            'corpus': 'm112en_fine',
            'feature_set': features_all_but_vector,
            'relation_set': "FULL_RELATION_SET",
            'segmentation': 'edu',
            'optimize': True,
            'optimize_weighting': False
        },
        'm112en-diss-edu-simple-op|train': {
            'corpus': 'm112en_fine',
            'feature_set': features_all_but_vector,
            'relation_set': "SIMPLE_RELATION_SET",
            'segmentation': 'edu',
            'optimize': True,
            'optimize_weighting': 'train'
        },
        'm112en-diss-edu-full-op|train': {
            'corpus': 'm112en_fine',
            'feature_set': features_all_but_vector,
            'relation_set': "FULL_RELATION_SET",
            'segmentation': 'edu',
            'optimize': True,
            'optimize_weighting': 'train'
        },
        'm112en-diss-edu-simple-op|test': {
            'corpus': 'm112en_fine',
            'feature_set': features_all_but_vector,
            'relation_set': "SIMPLE_RELATION_SET",
            'segmentation': 'edu',
            'optimize': True,
            'optimize_weighting': 'inner_cv'
        },
        'm112en-diss-edu-full-op|test': {
            'corpus': 'm112en_fine',
            'feature_set': features_all_but_vector,
            'relation_set': "FULL_RELATION_SET",
            'segmentation': 'edu',
            'optimize': True,
            'optimize_weighting': 'inner_cv'
        }
    }

    # run all experiment conditions
    folds = list(get_static_folds())
    for condition_name, params in conditions.items():
        print "### Running experiment condition", condition_name

        # load corpus
        corpus_name = params.pop('corpus')
        corpus = GraphCorpus()
        corpus.load(CORPORA[corpus_name]['path'])

        # set condition params
        language = CORPORA[corpus_name]['language']
        features = init_language(language)
        features.feature_set = params.pop('feature_set')
        params['relation_set'] = RELATION_SETS_BY_NAME.get(
            params['relation_set'])
        texts, trees = corpus.segments_trees(
            params.pop('segmentation'), params['relation_set'])

        # run condition
        predictions, _decisions = folds_static(
            texts, trees, folds, features, params, condition_name)
        with open('data/{}.json'.format(condition_name), 'w') as f:
            json.dump(predictions, f, indent=1, sort_keys=True)
